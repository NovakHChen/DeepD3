#!/usr/bin/env python3
"""Two-stage fine-tuning for DeepD3 model zoo weights."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from deepd3.training.stream import DataGeneratorStream

from common import D3SET_DIR, FINETUNED_MODEL_DIR, PRETRAINED_MODEL_DIR, ensure_dirs


def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    denom = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    dice = (2.0 * intersection + 1.0) / (denom + 1.0)
    return 1.0 - dice


def iou_score(y_true, y_pred):
    y_true = tf.cast(y_true > 0.5, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) - intersection
    return tf.reduce_mean((intersection + 1.0) / (union + 1.0))


def set_encoder_trainable(model: tf.keras.Model, trainable: bool) -> None:
    for layer in model.layers:
        if layer.name.startswith("enc_") or layer.name.startswith("latent"):
            layer.trainable = trainable
        else:
            layer.trainable = True


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=[dice_loss, "mse"],
        metrics=["acc", iou_score],
    )


def require_gpu(allow_cpu: bool) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus and not allow_cpu:
        raise RuntimeError("No TensorFlow GPU detected. Run gpu_preflight.py before training.")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def build_generators(args: argparse.Namespace, batch_size: int):
    target_resolution = None if args.target_resolution < 0 else args.target_resolution
    train = DataGeneratorStream(
        str(args.train_d3set),
        batch_size=batch_size,
        samples_per_epoch=args.samples_per_epoch,
        size=(args.tile_size, args.tile_size),
        target_resolution=target_resolution,
        min_content=args.min_content,
        augment=True,
        seed=args.seed,
    )
    validation = DataGeneratorStream(
        str(args.val_d3set),
        batch_size=batch_size,
        samples_per_epoch=args.validation_samples,
        size=(args.tile_size, args.tile_size),
        target_resolution=target_resolution,
        min_content=args.min_content,
        augment=False,
        shuffle=False,
        seed=args.seed,
    )
    return train, validation


def run_training(args: argparse.Namespace, batch_size: int) -> int:
    train_gen, val_gen = build_generators(args, batch_size)
    model = load_model(args.model, compile=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_encoder_trainable(model, False)
    compile_model(model, args.stage1_lr)
    stage1_callbacks = [
        ModelCheckpoint(str(args.output_dir / "stage1_best.h5"), monitor="val_loss", save_best_only=True),
        CSVLogger(str(args.output_dir / "stage1_history.csv")),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]
    model.fit(
        train_gen,
        epochs=args.stage1_epochs,
        validation_data=val_gen,
        callbacks=stage1_callbacks,
    )

    set_encoder_trainable(model, True)
    compile_model(model, args.stage2_lr)
    stage2_callbacks = [
        ModelCheckpoint(str(args.output_dir / "best.h5"), monitor="val_loss", save_best_only=True),
        CSVLogger(str(args.output_dir / "stage2_history.csv")),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]
    model.fit(
        train_gen,
        epochs=args.stage2_epochs,
        validation_data=val_gen,
        callbacks=stage2_callbacks,
    )
    final_path = args.output_dir / "final.h5"
    model.save(final_path)
    print(f"Saved {final_path}")
    return batch_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["invivo", "confocal"], required=True)
    parser.add_argument("--model", type=Path, default=PRETRAINED_MODEL_DIR / "DeepD3_16F.h5")
    parser.add_argument("--train-d3set", type=Path)
    parser.add_argument("--val-d3set", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--samples-per-epoch", type=int, default=8192)
    parser.add_argument("--validation-samples", type=int, default=2048)
    parser.add_argument("--min-content", type=float, default=50)
    parser.add_argument("--target-resolution", type=float, default=-1, help="-1 means None/mixed resolution.")
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--stage2-epochs", type=int, default=25)
    parser.add_argument("--stage1-lr", type=float, default=5e-4)
    parser.add_argument("--stage2-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()
    require_gpu(args.allow_cpu)
    args.train_d3set = args.train_d3set or (D3SET_DIR / f"{args.dataset}_train.d3set")
    args.val_d3set = args.val_d3set or (D3SET_DIR / f"{args.dataset}_heldout.d3set")
    args.output_dir = args.output_dir or (FINETUNED_MODEL_DIR / args.dataset)

    batch_size = args.batch_size
    while batch_size >= 1:
        try:
            used_batch_size = run_training(args, batch_size)
            print(f"Training completed with batch size {used_batch_size}")
            return
        except tf.errors.ResourceExhaustedError:
            print(f"GPU memory exhausted at batch size {batch_size}; retrying with smaller batch.")
            batch_size //= 2
            tf.keras.backend.clear_session()
            gc.collect()
    raise RuntimeError("Training failed even with batch size 1")


if __name__ == "__main__":
    main()
