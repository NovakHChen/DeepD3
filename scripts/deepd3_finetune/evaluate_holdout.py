#!/usr/bin/env python3
"""Run DeepD3 tiled heldout inference and compute label Dice/IoU."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import numpy as np
import tensorflow as tf
import tifffile
from tensorflow.keras.models import load_model

from common import DATASETS, FINETUNED_MODEL_DIR, PRETRAINED_MODEL_DIR, dice_iou, image_path_for_case, label_path_for_case


def parse_model_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name, Path(path)
    path = Path(value)
    return path.stem, path


def normalize_plane(plane: np.ndarray) -> np.ndarray:
    arr = plane.astype(np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)) * 2.0 - 1.0


def pad_to_shape(
    plane: np.ndarray,
    min_shape: tuple[int, int] = (0, 0),
    multiple: int = 32,
) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = plane.shape
    out_h = max(h, min_shape[0])
    out_w = max(w, min_shape[1])
    out_h += (multiple - (out_h % multiple)) % multiple
    out_w += (multiple - (out_w % multiple)) % multiple
    pad_h = out_h - h
    pad_w = out_w - w
    if pad_h or pad_w:
        mode = "reflect" if h > pad_h and w > pad_w else "edge"
        plane = np.pad(plane, ((0, pad_h), (0, pad_w)), mode=mode)
    return plane, (h, w)


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def predict_plane_tiled(
    model: tf.keras.Model,
    plane: np.ndarray,
    tile_size: int,
    tile_overlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = normalize_plane(plane)
    padded, original_shape = pad_to_shape(normalized, min_shape=(tile_size, tile_size))
    h, w = original_shape
    out_shape = padded.shape
    stride = tile_size - tile_overlap

    dendrite_accum = np.zeros(out_shape, dtype=np.float32)
    spine_accum = np.zeros(out_shape, dtype=np.float32)
    counts = np.zeros(out_shape, dtype=np.float32)

    for y in tile_starts(out_shape[0], tile_size, stride):
        for x in tile_starts(out_shape[1], tile_size, stride):
            tile = padded[y : y + tile_size, x : x + tile_size]
            pred_d, pred_s = model.predict(tile[None, ..., None], verbose=0)
            dendrite_accum[y : y + tile_size, x : x + tile_size] += pred_d.squeeze()
            spine_accum[y : y + tile_size, x : x + tile_size] += pred_s.squeeze()
            counts[y : y + tile_size, x : x + tile_size] += 1.0

    np.maximum(counts, 1.0, out=counts)
    return (dendrite_accum[:h, :w] / counts[:h, :w], spine_accum[:h, :w] / counts[:h, :w])


def predict_plane_full(model: tf.keras.Model, plane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized = normalize_plane(plane)
    padded, original_shape = pad_to_shape(normalized)
    pred_d, pred_s = model.predict(padded[None, ..., None], verbose=0)
    h, w = original_shape
    return pred_d.squeeze()[:h, :w], pred_s.squeeze()[:h, :w]


def predict_stack(
    model: tf.keras.Model,
    stack: np.ndarray,
    tile_size: int,
    tile_overlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    dendrites = np.zeros(stack.shape, dtype=np.float32)
    spines = np.zeros(stack.shape, dtype=np.float32)
    for z in range(stack.shape[0]):
        if tile_size > 0:
            pred_d, pred_s = predict_plane_tiled(model, stack[z], tile_size, tile_overlap)
        else:
            pred_d, pred_s = predict_plane_full(model, stack[z])
        dendrites[z] = pred_d
        spines[z] = pred_s
    return dendrites, spines


def combined_prediction(dendrites: np.ndarray, spines: np.ndarray, dendrite_threshold: float, spine_threshold: float) -> np.ndarray:
    out = np.zeros(dendrites.shape, dtype=np.uint8)
    out[dendrites >= dendrite_threshold] = 2
    out[spines >= spine_threshold] = 1
    return out


def evaluate_model(
    model_name: str,
    model_path: Path,
    dataset_key: str,
    out_dir: Path,
    dendrite_threshold: float,
    spine_threshold: float,
    tile_size: int,
    tile_overlap: int,
) -> dict:
    spec = DATASETS[dataset_key]
    dataset_dir = spec.target_dir
    cases = json.loads((dataset_dir / "split_manifest.json").read_text(encoding="utf-8"))["heldout_cases"]
    model = load_model(model_path, compile=False)
    model_out = out_dir / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for case_id in cases:
        image = tifffile.imread(image_path_for_case(dataset_dir, "heldout", case_id))
        ref = tifffile.imread(label_path_for_case(dataset_dir, "heldout", case_id))
        dendrites, spines = predict_stack(model, image, tile_size, tile_overlap)
        pred = combined_prediction(dendrites, spines, dendrite_threshold, spine_threshold)
        tifffile.imwrite(model_out / f"{case_id}.tif", pred)

        for label_name, label_value in [("spine", 1), ("dendrite", 2)]:
            metrics = dice_iou(pred == label_value, ref == label_value)
            rows.append(
                {
                    "model": model_name,
                    "case": case_id,
                    "label": label_name,
                    "label_value": label_value,
                    **metrics,
                }
            )

    summary = {}
    for label_name in ["spine", "dendrite"]:
        label_rows = [r for r in rows if r["label"] == label_name]
        summary[label_name] = {
            "Dice": float(np.mean([r["Dice"] for r in label_rows])),
            "IoU": float(np.mean([r["IoU"] for r in label_rows])),
            "n_cases": len(label_rows),
        }
    return {"model": model_name, "model_path": str(model_path), "rows": rows, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["invivo", "confocal"], required=True)
    parser.add_argument(
        "--model",
        action="append",
        help="Model as name=/path/to/model.h5. Can be repeated.",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--dendrite-threshold", type=float, default=0.7)
    parser.add_argument("--spine-threshold", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int, default=512, help="Use 0 for full-plane inference.")
    parser.add_argument("--tile-overlap", type=int, default=64)
    args = parser.parse_args()
    if args.tile_size < 0:
        raise ValueError("--tile-size must be >= 0")
    if args.tile_size and args.tile_size % 32:
        raise ValueError("--tile-size must be a multiple of 32")
    if args.tile_size and not (0 <= args.tile_overlap < args.tile_size):
        raise ValueError("--tile-overlap must be >= 0 and smaller than --tile-size")

    default_models = [
        ("pretrained_16F", PRETRAINED_MODEL_DIR / "DeepD3_16F.h5"),
        ("finetuned_16F", FINETUNED_MODEL_DIR / args.dataset / "best.h5"),
    ]
    models = [parse_model_arg(value) for value in args.model] if args.model else default_models
    out_dir = args.out_dir or (FINETUNED_MODEL_DIR / args.dataset / "heldout_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    model_summaries = {}
    for model_name, model_path in models:
        result = evaluate_model(
            model_name,
            model_path,
            args.dataset,
            out_dir,
            args.dendrite_threshold,
            args.spine_threshold,
            args.tile_size,
            args.tile_overlap,
        )
        all_rows.extend(result["rows"])
        model_summaries[model_name] = result["summary"]

    csv_path = out_dir / "holdout_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "case", "label", "label_value", "Dice", "IoU", "TP", "FP", "FN", "n_pred", "n_ref"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = out_dir / "holdout_summary.json"
    summary_path.write_text(json.dumps(model_summaries, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(model_summaries, indent=2))
    print(csv_path)
    print(summary_path)


if __name__ == "__main__":
    main()
