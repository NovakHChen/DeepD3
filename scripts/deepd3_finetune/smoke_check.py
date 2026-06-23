#!/usr/bin/env python3
"""Smoke-check model loading, d3set sampling, and one GPU train batch."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorflow.keras.models import load_model

from deepd3.training.stream import DataGeneratorStream

from common import D3SET_DIR, PRETRAINED_MODEL_DIR
from train_deepd3_finetune import compile_model, require_gpu


def check_model(path: Path) -> None:
    model = load_model(path, compile=False)
    print(
        f"model={path} input={model.input_shape} "
        f"outputs={[tuple(o.shape) for o in model.outputs]} params={model.count_params()}"
    )


def check_d3set(path: Path, batch_size: int) -> tuple:
    gen = DataGeneratorStream(
        str(path),
        batch_size=batch_size,
        samples_per_epoch=batch_size * 2,
        size=(128, 128),
        target_resolution=None,
        min_content=50,
        augment=False,
        shuffle=False,
    )
    x, y = gen[0]
    print(
        f"d3set={path.name} X={x.shape}/{x.dtype} "
        f"dendrite={y[0].shape}/{y[0].dtype} spines={y[1].shape}/{y[1].dtype} "
        f"Xrange=({float(x.min()):.3f},{float(x.max()):.3f})"
    )
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-batch", action="store_true")
    args = parser.parse_args()

    require_gpu(args.allow_cpu)
    for model_path in [PRETRAINED_MODEL_DIR / "DeepD3_8F.h5", PRETRAINED_MODEL_DIR / "DeepD3_16F.h5"]:
        check_model(model_path)

    first_batch = None
    for d3set in sorted(D3SET_DIR.glob("*.d3set")):
        batch = check_d3set(d3set, args.batch_size)
        if first_batch is None and d3set.name == "invivo_train.d3set":
            first_batch = batch

    if args.train_batch:
        if first_batch is None:
            raise RuntimeError("No invivo_train.d3set batch was available for training smoke check")
        model = load_model(PRETRAINED_MODEL_DIR / "DeepD3_8F.h5", compile=False)
        compile_model(model, learning_rate=1e-5)
        metrics = model.train_on_batch(first_batch[0], first_batch[1], return_dict=True)
        print(f"one_batch_train={metrics}")


if __name__ == "__main__":
    main()
