#!/usr/bin/env python3
"""Render RESPAN-style heldout overlays and metrics for DeepD3 predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
import tifffile
from scipy import ndimage
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from common import DATASETS, FINETUNED_MODEL_DIR, dice_iou, image_path_for_case, label_path_for_case, read_json


LABELS = {"spine": 1, "dendrite": 2}
OVERLAY_COLORS = {
    "spine": np.array([1.0, 0.12, 0.12]),
    "dendrite": np.array([0.0, 0.85, 1.0]),
}
RAW_COLOR = "#555555"
FINE_COLOR = "#ff4f93"
LINE_COLOR = "#b6b6b6"
TEXT_COLOR = "#222222"
THRESHOLDS = np.round(np.linspace(0, 1, 41), 3)
DETECTION_THRESHOLD = 0.25
TRAPEZOID = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def read_stack(path: Path) -> np.ndarray:
    arr = np.asarray(tifffile.imread(path))
    if arr.ndim == 2:
        arr = arr[None]
    return arr


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def normalize_mip(raw: np.ndarray) -> np.ndarray:
    mip = raw.max(axis=0).astype(np.float32)
    lo, hi = np.percentile(mip, [1, 99.7])
    if hi <= lo:
        lo, hi = float(mip.min()), float(mip.max())
    if hi <= lo:
        return np.zeros_like(mip, dtype=np.float32)
    return np.clip((mip - lo) / (hi - lo), 0, 1)


def label_mips(mask: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.any(mask == value, axis=0) for name, value in LABELS.items()}


def overlay_on_base(base: np.ndarray, mask: np.ndarray, alpha: float = 0.58) -> np.ndarray:
    rgb = np.stack([base, base, base], axis=-1)
    for label_name, color in OVERLAY_COLORS.items():
        label_mask = label_mips(mask)[label_name]
        rgb[label_mask] = (1 - alpha) * rgb[label_mask] + alpha * color
    return rgb


def load_case_images(dataset_dir: Path, pred_root: Path, case: str) -> dict[str, np.ndarray]:
    raw = read_stack(image_path_for_case(dataset_dir, "heldout", case))
    manual = read_stack(label_path_for_case(dataset_dir, "heldout", case))
    pretrained = read_stack(pred_root / "pretrained_16F" / f"{case}.tif")
    finetuned = read_stack(pred_root / "finetuned_16F" / f"{case}.tif")
    shapes = {arr.shape for arr in [raw, manual, pretrained, finetuned]}
    if len(shapes) != 1:
        raise ValueError(f"{case}: shape mismatch {shapes}")
    base = normalize_mip(raw)
    return {
        "raw": np.stack([base, base, base], axis=-1),
        "manual": overlay_on_base(base, manual),
        "pretrained": overlay_on_base(base, pretrained),
        "finetuned": overlay_on_base(base, finetuned),
    }


def add_overlay_legend(fig) -> None:
    handles = [
        Patch(facecolor=OVERLAY_COLORS["spine"], label="spine"),
        Patch(facecolor=OVERLAY_COLORS["dendrite"], label="dendrite"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=11)


def render_overlay_grid(
    dataset_dir: Path,
    pred_root: Path,
    cases: list[str],
    columns: list[tuple[str, str]],
    out_path: Path,
    title: str,
    fig_width: float,
) -> None:
    fig, axes = plt.subplots(
        len(cases),
        len(columns),
        figsize=(fig_width, 3.7 * len(cases)),
        constrained_layout=False,
    )
    if len(cases) == 1:
        axes = axes[None, :]
    if len(columns) == 1:
        axes = axes[:, None]

    for row_idx, case in enumerate(cases):
        images = load_case_images(dataset_dir, pred_root, case)
        for col_idx, (key, column_title) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            ax.imshow(images[key], interpolation="nearest")
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(column_title, fontsize=12, pad=8)
            if col_idx == 0:
                ax.text(
                    -0.04,
                    0.5,
                    case,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=11,
                )

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    add_overlay_legend(fig)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.965, bottom=0.035, wspace=0.04, hspace=0.08)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_overlays(dataset_dir: Path, pred_root: Path, cases: list[str], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    four_columns = [
        ("raw", "Raw TIFF MIP"),
        ("manual", "Manual annotation overlay"),
        ("pretrained", "Pretrained 16F overlay"),
        ("finetuned", "Fine-tuned 16F overlay"),
    ]
    manual_vs_fine = [
        ("manual", "Manual annotation overlay"),
        ("finetuned", "Fine-tuned 16F overlay"),
    ]

    outputs.append(out_dir / "heldout5_raw_manual_pretrained_finetuned_overlay.png")
    render_overlay_grid(
        dataset_dir,
        pred_root,
        cases,
        four_columns,
        outputs[-1],
        "Held-out stacks: raw, manual annotation, pretrained DeepD3, fine-tuned DeepD3",
        fig_width=16.5,
    )
    outputs.append(out_dir / "heldout5_manual_vs_finetuned_overlay.png")
    render_overlay_grid(
        dataset_dir,
        pred_root,
        cases,
        manual_vs_fine,
        outputs[-1],
        "Held-out stacks: manual annotation vs fine-tuned DeepD3",
        fig_width=8.8,
    )

    for case in cases:
        outputs.append(out_dir / f"{case}_raw_manual_pretrained_finetuned_overlay.png")
        render_overlay_grid(dataset_dir, pred_root, [case], four_columns, outputs[-1], case, fig_width=14.5)
        outputs.append(out_dir / f"{case}_manual_vs_finetuned_overlay.png")
        render_overlay_grid(dataset_dir, pred_root, [case], manual_vs_fine, outputs[-1], case, fig_width=7.8)
    return outputs


def compute_binary_metrics(pred: np.ndarray, ref: np.ndarray, label: int) -> dict:
    pred_mask = pred == label
    ref_mask = ref == label
    tp = int(np.logical_and(pred_mask, ref_mask).sum())
    fp = int(np.logical_and(pred_mask, ~ref_mask).sum())
    fn = int(np.logical_and(~pred_mask, ref_mask).sum())
    tn = int(np.logical_and(~pred_mask, ~ref_mask).sum())
    n_pred = tp + fp
    n_ref = tp + fn
    total = tp + fp + fn + tn
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "n_pred": n_pred,
        "n_ref": n_ref,
        "Dice_F1": safe_div(2 * tp, 2 * tp + fp + fn),
        "IoU_Jaccard": safe_div(tp, tp + fp + fn),
        "Precision": safe_div(tp, tp + fp),
        "Recall_Sensitivity": safe_div(tp, tp + fn),
        "Specificity": safe_div(tn, tn + fp),
        "Accuracy": safe_div(tp + tn, total),
        "False_Positive_Rate": safe_div(fp, fp + tn),
        "False_Negative_Rate": safe_div(fn, fn + tp),
        "Volume_Similarity": 1.0 - safe_div(abs(fn - fp), 2 * tp + fp + fn),
        "Predicted_to_GT_Volume_Ratio": safe_div(n_pred, n_ref),
    }


def mean(records: list[dict], key: str) -> float:
    return float(np.mean([r[key] for r in records]))


def aggregate_counts(records: list[dict]) -> dict:
    counts = {key: int(sum(r[key] for r in records)) for key in ["TP", "FP", "FN", "TN", "n_pred", "n_ref"]}
    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    counts.update(
        {
            "Dice_F1": safe_div(2 * tp, 2 * tp + fp + fn),
            "IoU_Jaccard": safe_div(tp, tp + fp + fn),
            "Precision": safe_div(tp, tp + fp),
            "Recall_Sensitivity": safe_div(tp, tp + fn),
            "Specificity": safe_div(tn, tn + fp),
            "Accuracy": safe_div(tp + tn, tp + fp + fn + tn),
            "False_Positive_Rate": safe_div(fp, fp + tn),
            "False_Negative_Rate": safe_div(fn, fn + tp),
            "Volume_Similarity": 1.0 - safe_div(abs(fn - fp), 2 * tp + fp + fn),
            "Predicted_to_GT_Volume_Ratio": safe_div(tp + fp, tp + fn),
        }
    )
    return counts


def compute_respan_metrics(dataset_dir: Path, pred_root: Path, cases: list[str], out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = {
        "original_pretrained": pred_root / "pretrained_16F",
        "finetuned_holdout5": pred_root / "finetuned_16F",
    }
    rows = []
    for case in cases:
        ref = read_stack(label_path_for_case(dataset_dir, "heldout", case))
        for model_name, pred_dir in predictions.items():
            pred_path = pred_dir / f"{case}.tif"
            pred = read_stack(pred_path)
            if pred.shape != ref.shape:
                raise ValueError(f"{case} {model_name}: prediction {pred.shape} != GT {ref.shape}")
            for label_name, label_value in LABELS.items():
                rows.append(
                    {
                        "case": case,
                        "model": model_name,
                        "label": label_name,
                        "label_value": label_value,
                        **compute_binary_metrics(pred, ref, label_value),
                    }
                )

    fieldnames = [
        "case",
        "model",
        "label",
        "label_value",
        "TP",
        "FP",
        "FN",
        "TN",
        "n_pred",
        "n_ref",
        "Dice_F1",
        "IoU_Jaccard",
        "Precision",
        "Recall_Sensitivity",
        "Specificity",
        "Accuracy",
        "False_Positive_Rate",
        "False_Negative_Rate",
        "Volume_Similarity",
        "Predicted_to_GT_Volume_Ratio",
    ]
    by_case_csv = out_dir / "respan_style_metrics_by_case_label.csv"
    with by_case_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metric_fields = fieldnames[10:]
    summary_rows = []
    for model_name in predictions:
        model_rows = [r for r in rows if r["model"] == model_name]
        for label_name in LABELS:
            label_rows = [r for r in model_rows if r["label"] == label_name]
            macro = {field: mean(label_rows, field) for field in metric_fields}
            micro = aggregate_counts(label_rows)
            summary_rows.append(
                {
                    "model": model_name,
                    "label_group": label_name,
                    "n_cases": len(label_rows),
                    "aggregation": "macro_case_mean",
                    **{k: "" for k in ["TP", "FP", "FN", "TN", "n_pred", "n_ref"]},
                    **macro,
                }
            )
            summary_rows.append(
                {
                    "model": model_name,
                    "label_group": label_name,
                    "n_cases": len(label_rows),
                    "aggregation": "micro_voxel_total",
                    **{k: micro[k] for k in ["TP", "FP", "FN", "TN", "n_pred", "n_ref"]},
                    **{field: micro[field] for field in metric_fields},
                }
            )

        macro = {field: mean(model_rows, field) for field in metric_fields}
        micro = aggregate_counts(model_rows)
        summary_rows.append(
            {
                "model": model_name,
                "label_group": "foreground",
                "n_cases": len(model_rows),
                "aggregation": "macro_case_label_mean",
                **{k: "" for k in ["TP", "FP", "FN", "TN", "n_pred", "n_ref"]},
                **macro,
            }
        )
        summary_rows.append(
            {
                "model": model_name,
                "label_group": "foreground",
                "n_cases": len(model_rows),
                "aggregation": "micro_voxel_total",
                **{k: micro[k] for k in ["TP", "FP", "FN", "TN", "n_pred", "n_ref"]},
                **{field: micro[field] for field in metric_fields},
            }
        )

    summary_fieldnames = [
        "model",
        "label_group",
        "n_cases",
        "aggregation",
        "TP",
        "FP",
        "FN",
        "TN",
        "n_pred",
        "n_ref",
        *metric_fields,
    ]
    summary_csv = out_dir / "respan_style_metrics_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = out_dir / "respan_style_metrics_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "gt_dir": str(dataset_dir / "labelsTs"),
                "prediction_dirs": {k: str(v) for k, v in predictions.items()},
                "metrics": {
                    "Dice_F1": "2TP / (2TP + FP + FN)",
                    "IoU_Jaccard": "TP / (TP + FP + FN)",
                    "Precision": "TP / (TP + FP)",
                    "Recall_Sensitivity": "TP / (TP + FN)",
                    "Specificity": "TN / (TN + FP)",
                    "Accuracy": "(TP + TN) / total voxels",
                    "Volume_Similarity": "1 - abs(FN - FP) / (2TP + FP + FN)",
                },
                "summary_rows": summary_rows,
                "by_case_label": rows,
            },
            indent=4,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary_csv, by_case_csv, summary_json


def load_summary(summary_csv: Path) -> list[dict]:
    with summary_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summary_get(rows: list[dict], model: str, label: str, metric: str) -> float:
    agg = "macro_case_label_mean" if label == "foreground" else "macro_case_mean"
    row = next(r for r in rows if r["model"] == model and r["label_group"] == label and r["aggregation"] == agg)
    return float(row[metric])


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def render_metric_comparison(summary_csv: Path, out_path: Path, dataset_title: str) -> Path:
    rows = load_summary(summary_csv)
    metrics = [
        ("Dice_F1", "Dice/F1"),
        ("IoU_Jaccard", "IoU/Jaccard"),
        ("Precision", "Precision"),
        ("Recall_Sensitivity", "Recall/Sensitivity"),
        ("Volume_Similarity", "Volume similarity"),
    ]
    labels = ["spine", "dendrite", "foreground"]
    display = {"spine": "Spine", "dendrite": "Dendrite", "foreground": "Foreground mean"}
    colors = {"original_pretrained": "#7a8798", "finetuned_holdout5": "#1f9d8a"}
    fig, axes = plt.subplots(len(labels), 1, figsize=(13, 10), constrained_layout=True)
    x = np.arange(len(metrics))
    width = 0.36

    for ax, label in zip(axes, labels):
        original = [summary_get(rows, "original_pretrained", label, metric) for metric, _ in metrics]
        finetuned = [summary_get(rows, "finetuned_holdout5", label, metric) for metric, _ in metrics]
        ax.bar(x - width / 2, original, width, color=colors["original_pretrained"], label="Original pretrained")
        ax.bar(x + width / 2, finetuned, width, color=colors["finetuned_holdout5"], label="Fine-tuned")
        for i, (orig, fine) in enumerate(zip(original, finetuned)):
            delta = fine - orig
            color = "#1f9d8a" if delta >= 0 else "#c94f4f"
            ax.text(i, max(orig, fine) + 0.045, f"{delta:+.3f}", ha="center", fontsize=9, color=color)
        ax.set_ylim(0, 1.12)
        ax.set_xticks(x)
        ax.set_xticklabels([name for _, name in metrics])
        ax.set_ylabel("Score")
        ax.set_title(display[label])
        style_axis(ax)

    axes[0].legend(loc="lower right", frameon=False)
    fig.suptitle(f"{dataset_title}: Held-out GT Evaluation, DeepD3 16F vs Fine-tuned 16F", fontsize=15)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def component_pairs(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    structure = np.ones((3, 3, 3), dtype=bool)
    gt_lab, n_gt = ndimage.label(gt_mask, structure=structure)
    pred_lab, n_pred = ndimage.label(pred_mask, structure=structure)
    gt_vol = np.bincount(gt_lab.ravel(), minlength=n_gt + 1)
    pred_vol = np.bincount(pred_lab.ravel(), minlength=n_pred + 1)

    pairs = []
    overlap = np.logical_and(gt_lab > 0, pred_lab > 0)
    if overlap.any():
        encoded = gt_lab[overlap].astype(np.int64) * (n_pred + 1) + pred_lab[overlap].astype(np.int64)
        unique, inter = np.unique(encoded, return_counts=True)
        for code, intersection in zip(unique, inter):
            gt_id = int(code // (n_pred + 1))
            pred_id = int(code % (n_pred + 1))
            union = int(gt_vol[gt_id] + pred_vol[pred_id] - intersection)
            pairs.append(
                {
                    "gt_id": gt_id,
                    "pred_id": pred_id,
                    "iou": safe_div(int(intersection), union),
                    "intersection": int(intersection),
                }
            )
    return {
        "gt_lab": gt_lab,
        "pred_lab": pred_lab,
        "n_gt": int(n_gt),
        "n_pred": int(n_pred),
        "pairs": pairs,
        "gt_slices": ndimage.find_objects(gt_lab),
        "pred_slices": ndimage.find_objects(pred_lab),
    }


def greedy_match(n_gt: int, n_pred: int, pairs: list[dict], threshold: float) -> dict:
    matched_gt = set()
    matched_pred = set()
    matched_ious = []
    for pair in sorted(pairs, key=lambda x: x["iou"], reverse=True):
        if pair["iou"] < threshold:
            break
        gt_id = pair["gt_id"]
        pred_id = pair["pred_id"]
        if gt_id in matched_gt or pred_id in matched_pred:
            continue
        matched_gt.add(gt_id)
        matched_pred.add(pred_id)
        matched_ious.append(pair["iou"])
    tp = len(matched_gt)
    fp = n_pred - tp
    fn = n_gt - tp
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
    }


def best_iou_by_gt(n_gt: int, pairs: list[dict]) -> tuple[list[float], dict[int, int]]:
    best = {gt_id: (0.0, 0) for gt_id in range(1, n_gt + 1)}
    for pair in pairs:
        gt_id = pair["gt_id"]
        if pair["iou"] > best[gt_id][0]:
            best[gt_id] = (pair["iou"], pair["pred_id"])
    return [best[gt_id][0] for gt_id in range(1, n_gt + 1)], {
        gt_id: pred_id for gt_id, (iou, pred_id) in best.items() if iou > 0 and pred_id > 0
    }


def union_slices(slice_a, slice_b, shape: tuple[int, ...], pad: int = 1) -> tuple[slice, slice, slice]:
    result = []
    for dim, (sa, sb) in enumerate(zip(slice_a, slice_b)):
        start = max(min(sa.start, sb.start) - pad, 0)
        stop = min(max(sa.stop, sb.stop) + pad, shape[dim])
        result.append(slice(start, stop))
    return tuple(result)


def surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    return np.logical_and(mask, ~eroded)


def hausdorff_distance(gt_lab: np.ndarray, pred_lab: np.ndarray, gt_slices, pred_slices, gt_id: int, pred_id: int) -> float:
    gt_slice = gt_slices[gt_id - 1]
    pred_slice = pred_slices[pred_id - 1]
    if gt_slice is None or pred_slice is None:
        return float("nan")
    bbox = union_slices(gt_slice, pred_slice, gt_lab.shape)
    gt_mask = gt_lab[bbox] == gt_id
    pred_mask = pred_lab[bbox] == pred_id
    gt_coords = np.argwhere(surface(gt_mask))
    pred_coords = np.argwhere(surface(pred_mask))
    if gt_coords.size == 0 or pred_coords.size == 0:
        return float("nan")
    gt_to_pred = cKDTree(pred_coords).query(gt_coords, k=1)[0]
    pred_to_gt = cKDTree(gt_coords).query(pred_coords, k=1)[0]
    return float(max(gt_to_pred.max(), pred_to_gt.max()))


def collect_spine_object_data(dataset_dir: Path, pred_root: Path, cases: list[str]) -> dict:
    pred_dirs = {
        "raw_model": pred_root / "pretrained_16F",
        "fine_tuned": pred_root / "finetuned_16F",
    }
    voxel_rows = []
    object_rows = []
    best_iou = {model: [] for model in pred_dirs}
    hd = {model: [] for model in pred_dirs}
    f1_curves = {model: np.zeros_like(THRESHOLDS, dtype=float) for model in pred_dirs}
    f1_case_auc = {model: [] for model in pred_dirs}
    agreement_masks = {"GT": [], "Raw model": [], "Fine-tuned": []}

    for case in cases:
        gt = read_stack(label_path_for_case(dataset_dir, "heldout", case))
        agreement_masks["GT"].append(gt == LABELS["spine"])

        for model, pred_dir in pred_dirs.items():
            pred = read_stack(pred_dir / f"{case}.tif")
            metrics = dice_iou(pred == LABELS["spine"], gt == LABELS["spine"])
            voxel_rows.append(
                {
                    "case": case,
                    "model": model,
                    "dice": metrics["Dice"],
                    "iou": metrics["IoU"],
                    "precision": safe_div(metrics["TP"], metrics["TP"] + metrics["FP"]),
                    "recall": safe_div(metrics["TP"], metrics["TP"] + metrics["FN"]),
                }
            )
            agreement_masks["Raw model" if model == "raw_model" else "Fine-tuned"].append(pred == LABELS["spine"])

            comp = component_pairs(pred == LABELS["spine"], gt == LABELS["spine"])
            best_vals, best_ids = best_iou_by_gt(comp["n_gt"], comp["pairs"])
            best_iou[model].extend(best_vals)

            for gt_id, pred_id in best_ids.items():
                dist = hausdorff_distance(comp["gt_lab"], comp["pred_lab"], comp["gt_slices"], comp["pred_slices"], gt_id, pred_id)
                if np.isfinite(dist):
                    hd[model].append(dist)

            det = greedy_match(comp["n_gt"], comp["n_pred"], comp["pairs"], DETECTION_THRESHOLD)
            object_rows.append(
                {
                    "case": case,
                    "model": model,
                    "threshold": DETECTION_THRESHOLD,
                    "gt_objects": comp["n_gt"],
                    "pred_objects": comp["n_pred"],
                    "true_positive_rate": det["recall"],
                    "false_positive_rate": safe_div(det["fp"], det["tp"] + det["fp"]),
                    "false_negative_rate": safe_div(det["fn"], det["tp"] + det["fn"]),
                    "object_precision": det["precision"],
                    "object_recall": det["recall"],
                    "object_f1": det["f1"],
                    "matched_iou_mean": det["mean_iou"],
                    "tp": det["tp"],
                    "fp": det["fp"],
                    "fn": det["fn"],
                }
            )

            per_threshold = []
            for i, threshold in enumerate(THRESHOLDS):
                match = greedy_match(comp["n_gt"], comp["n_pred"], comp["pairs"], float(threshold))
                f1_curves[model][i] += match["f1"] / len(cases)
                per_threshold.append(match["f1"])
            f1_case_auc[model].append(float(TRAPEZOID(per_threshold, THRESHOLDS)))

    agreement = {}
    for name_a, masks_a in agreement_masks.items():
        agreement[name_a] = {}
        a = np.concatenate([m.ravel() for m in masks_a])
        for name_b, masks_b in agreement_masks.items():
            b = np.concatenate([m.ravel() for m in masks_b])
            inter = int(np.logical_and(a, b).sum())
            agreement[name_a][name_b] = safe_div(2 * inter, int(a.sum()) + int(b.sum()))

    return {
        "cases": cases,
        "voxel_rows": voxel_rows,
        "object_rows": object_rows,
        "best_iou": best_iou,
        "hausdorff": hd,
        "f1_curves": f1_curves,
        "f1_case_auc": f1_case_auc,
        "agreement": agreement,
    }


def sem(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals, ddof=1) / math.sqrt(len(vals)))


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)


def paired_box(ax, raw_vals: list[float], fine_vals: list[float], ylabel: str, title: str, letter: str) -> None:
    bp = ax.boxplot(
        [raw_vals, fine_vals],
        positions=[1, 2],
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
        boxprops={"linewidth": 1.2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )
    for patch, color in zip(bp["boxes"], [RAW_COLOR, FINE_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    for i, (raw, fine) in enumerate(zip(raw_vals, fine_vals)):
        jitter = (i - (len(raw_vals) - 1) / 2) * 0.025
        ax.plot([1 + jitter, 2 + jitter], [raw, fine], color=LINE_COLOR, linewidth=1.0, zorder=1)
        ax.scatter(1 + jitter, raw, color=RAW_COLOR, s=22, zorder=2)
        ax.scatter(2 + jitter, fine, color=FINE_COLOR, s=22, zorder=2)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Raw model", "Fine-tuned"], rotation=35, ha="right")
    ax.set_ylim(0, 1.03)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold")
    delta = float(np.mean(fine_vals) - np.mean(raw_vals))
    ax.text(
        0.98,
        0.93,
        f"$\\Delta$={delta:+.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=TEXT_COLOR,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
    )
    clean_axis(ax)


def plot_agreement(ax, agreement: dict) -> None:
    names = ["GT", "Raw model", "Fine-tuned"]
    data = np.array([[agreement[a][b] for b in names] for a in names])
    cmap = LinearSegmentedColormap.from_list("respan_agreement", ["#d92b7f", "#ff77ad", "#f3e74f"])
    ax.imshow(data, vmin=0, vmax=1, cmap=cmap)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=9, color="black")
    ax.set_title("A  Spine agreement", loc="left", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_detection_rates(ax, object_rows: list[dict]) -> None:
    groups = [
        ("true_positive_rate", "True positive"),
        ("false_positive_rate", "False positive"),
        ("false_negative_rate", "False negative"),
    ]
    centers = np.arange(len(groups)) + 1
    offset = 0.17
    tick_labels = []
    for idx, (key, label) in enumerate(groups):
        raw = [r[key] for r in sorted(object_rows, key=lambda x: x["case"]) if r["model"] == "raw_model"]
        fine = [r[key] for r in sorted(object_rows, key=lambda x: x["case"]) if r["model"] == "fine_tuned"]
        tick_labels.append(f"{label}\n$\\Delta$={float(np.mean(fine) - np.mean(raw)):+.2f}")
        x0, x1 = centers[idx] - offset, centers[idx] + offset
        for i, (rv, fv) in enumerate(zip(raw, fine)):
            jitter = (i - (len(raw) - 1) / 2) * 0.015
            ax.plot([x0 + jitter, x1 + jitter], [rv, fv], color=LINE_COLOR, linewidth=0.9, zorder=1)
            ax.scatter(x0 + jitter, rv, color=RAW_COLOR, s=18, zorder=2)
            ax.scatter(x1 + jitter, fv, color=FINE_COLOR, s=18, zorder=2)
        ax.errorbar(x0, np.mean(raw), yerr=sem(raw), color="black", marker="_", markersize=14, capsize=3, linewidth=1.2, zorder=3)
        ax.errorbar(x1, np.mean(fine), yerr=sem(fine), color="black", marker="_", markersize=14, capsize=3, linewidth=1.2, zorder=3)
    ax.set_xticks(centers)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right")
    ax.tick_params(axis="x", labelsize=7)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Detection rate")
    ax.set_title("D  Spine detection rate (IoU >= 0.25)", loc="left", fontweight="bold")
    clean_axis(ax)


def plot_mask_iou(ax, best_iou: dict) -> None:
    rng = np.random.default_rng(7)
    raw = np.asarray(best_iou["raw_model"], dtype=float)
    fine = np.asarray(best_iou["fine_tuned"], dtype=float)
    for pos, vals, color in [(1, raw, RAW_COLOR), (2, fine, FINE_COLOR)]:
        jitter = rng.normal(0, 0.045, len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=9, color=color, alpha=0.45, linewidths=0)
        ax.errorbar(pos, np.mean(vals), yerr=sem(list(vals)), color="black", marker="_", markersize=18, capsize=4, linewidth=1.3)
    bp = ax.boxplot([raw, fine], positions=[1, 2], widths=0.48, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [RAW_COLOR, FINE_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.18)
    for element in ["medians", "whiskers", "caps"]:
        for artist in bp[element]:
            artist.set_color("black")
            artist.set_linewidth(1.0)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Raw model", "Fine-tuned"], rotation=35, ha="right")
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Spine mask IoU")
    ax.set_title("E  Spine mask IoU", loc="left", fontweight="bold")
    ax.text(
        0.98,
        0.93,
        f"$\\Delta$={float(np.mean(fine) - np.mean(raw)):+.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=TEXT_COLOR,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
    )
    clean_axis(ax)


def plot_f1_curve(ax, f1_curves: dict, f1_case_auc: dict) -> None:
    ax.plot(THRESHOLDS, f1_curves["raw_model"], color=RAW_COLOR, linewidth=2.0, label="Raw model")
    ax.plot(THRESHOLDS, f1_curves["fine_tuned"], color=FINE_COLOR, linewidth=2.0, label="Fine-tuned")
    delta_auc = float(np.mean(f1_case_auc["fine_tuned"]) - np.mean(f1_case_auc["raw_model"]))
    ax.text(
        0.98,
        0.93,
        f"$\\Delta$AUC={delta_auc:+.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=TEXT_COLOR,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("F1 score")
    ax.set_title("F  Object F1 across IoU thresholds", loc="left", fontweight="bold")
    clean_axis(ax)


def plot_hd_cdf(ax, hausdorff: dict) -> None:
    max_x = 0.0
    for model, color, label in [("raw_model", RAW_COLOR, "Raw model"), ("fine_tuned", FINE_COLOR, "Fine-tuned")]:
        vals = np.sort(np.asarray(hausdorff[model], dtype=float))
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, color=color, linewidth=2.0, label=label)
        max_x = max(max_x, float(np.percentile(vals, 95)))
    ax.set_xlim(0, max(4, max_x * 1.1))
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Hausdorff distance (voxels)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("G  Spine boundary distance", loc="left", fontweight="bold")
    raw = [v for v in hausdorff["raw_model"] if np.isfinite(v)]
    fine = [v for v in hausdorff["fine_tuned"] if np.isfinite(v)]
    if raw and fine:
        ax.text(
            0.98,
            0.32,
            f"$\\Delta$median={float(np.median(fine) - np.median(raw)):+.2f} vx",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color=TEXT_COLOR,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
        )
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax)


def write_object_outputs(data: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    object_out = out_dir / "spine_object_level_metrics.csv"
    with object_out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "case",
            "model",
            "threshold",
            "gt_objects",
            "pred_objects",
            "true_positive_rate",
            "false_positive_rate",
            "false_negative_rate",
            "object_precision",
            "object_recall",
            "object_f1",
            "matched_iou_mean",
            "tp",
            "fp",
            "fn",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data["object_rows"])

    raw_recall = [r["recall"] for r in data["voxel_rows"] if r["model"] == "raw_model"]
    fine_recall = [r["recall"] for r in data["voxel_rows"] if r["model"] == "fine_tuned"]
    raw_precision = [r["precision"] for r in data["voxel_rows"] if r["model"] == "raw_model"]
    fine_precision = [r["precision"] for r in data["voxel_rows"] if r["model"] == "fine_tuned"]
    raw_iou = np.asarray(data["best_iou"]["raw_model"], dtype=float)
    fine_iou = np.asarray(data["best_iou"]["fine_tuned"], dtype=float)
    rows = [
        ["n_heldout_cases", len(data["cases"])],
        ["n_gt_spine_objects", len(raw_iou)],
        ["voxel_recall_raw_mean", np.mean(raw_recall)],
        ["voxel_recall_finetuned_mean", np.mean(fine_recall)],
        ["voxel_precision_raw_mean", np.mean(raw_precision)],
        ["voxel_precision_finetuned_mean", np.mean(fine_precision)],
        ["spine_mask_iou_raw_mean", np.mean(raw_iou)],
        ["spine_mask_iou_finetuned_mean", np.mean(fine_iou)],
        ["f1_auc_raw_mean", np.mean(data["f1_case_auc"]["raw_model"])],
        ["f1_auc_finetuned_mean", np.mean(data["f1_case_auc"]["fine_tuned"])],
        ["hausdorff_raw_median", np.median(data["hausdorff"]["raw_model"]) if data["hausdorff"]["raw_model"] else float("nan")],
        [
            "hausdorff_finetuned_median",
            np.median(data["hausdorff"]["fine_tuned"]) if data["hausdorff"]["fine_tuned"] else float("nan"),
        ],
    ]
    summary_out = out_dir / "respan_paper_matched_holdout_summary.csv"
    with summary_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)
    return summary_out, object_out


def render_object_panel(data: dict, out_path: Path, dataset_title: str) -> Path:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(13.2, 7.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.05, 1.0, 1.0, 1.08], hspace=0.52, wspace=0.48)

    plot_agreement(fig.add_subplot(gs[0, 0]), data["agreement"])
    raw_recall = [r["recall"] for r in sorted(data["voxel_rows"], key=lambda x: x["case"]) if r["model"] == "raw_model"]
    fine_recall = [r["recall"] for r in sorted(data["voxel_rows"], key=lambda x: x["case"]) if r["model"] == "fine_tuned"]
    raw_precision = [r["precision"] for r in sorted(data["voxel_rows"], key=lambda x: x["case"]) if r["model"] == "raw_model"]
    fine_precision = [r["precision"] for r in sorted(data["voxel_rows"], key=lambda x: x["case"]) if r["model"] == "fine_tuned"]

    paired_box(fig.add_subplot(gs[0, 1]), raw_recall, fine_recall, "Recall", "Recall", "B")
    paired_box(fig.add_subplot(gs[0, 2]), raw_precision, fine_precision, "Precision", "Precision", "C")
    plot_detection_rates(fig.add_subplot(gs[0, 3]), data["object_rows"])
    plot_mask_iou(fig.add_subplot(gs[1, 0:2]), data["best_iou"])
    plot_f1_curve(fig.add_subplot(gs[1, 2]), data["f1_curves"], data["f1_case_auc"])
    plot_hd_cdf(fig.add_subplot(gs[1, 3]), data["hausdorff"])

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=RAW_COLOR, label="Raw model", markersize=7),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=FINE_COLOR, label="Fine-tuned model", markersize=7),
        plt.Line2D([0], [0], color=LINE_COLOR, linewidth=1, label="paired held-out stack"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{dataset_title}: Held-out Spine Segmentation Validation", fontsize=14, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.945,
        f"Pretrained DeepD3 16F vs fine-tuned DeepD3 16F; n={len(data['cases'])} held-out stacks; object metrics use 3D connected components, IoU threshold={DETECTION_THRESHOLD}.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out_path


def assert_predictions_exist(pred_root: Path, cases: list[str]) -> None:
    for model_dir in ["pretrained_16F", "finetuned_16F"]:
        for case in cases:
            pred_path = pred_root / model_dir / f"{case}.tif"
            if not pred_path.exists():
                raise FileNotFoundError(f"Missing prediction: {pred_path}")


def render_dataset(dataset_key: str) -> dict[str, list[str]]:
    spec = DATASETS[dataset_key]
    dataset_dir = spec.target_dir
    cases = read_json(dataset_dir / "split_manifest.json")["heldout_cases"]
    pred_root = FINETUNED_MODEL_DIR / dataset_key / "heldout_eval"
    assert_predictions_exist(pred_root, cases)

    dataset_title = "In-vivo" if dataset_key == "invivo" else "Confocal"
    evaluation_dir = FINETUNED_MODEL_DIR / dataset_key / "evaluation"
    overlay_dir = evaluation_dir / "overlay_figures" / "manual_model_overlays"
    metric_dir = evaluation_dir / "respan_style_metrics" / "metrics"
    paper_dir = evaluation_dir / "paper_matched_panels"

    overlay_outputs = render_overlays(dataset_dir, pred_root, cases, overlay_dir)
    summary_csv, by_case_csv, summary_json = compute_respan_metrics(dataset_dir, pred_root, cases, metric_dir)
    metric_plot = render_metric_comparison(summary_csv, metric_dir / "respan_style_metric_comparison.png", dataset_title)

    object_data = collect_spine_object_data(dataset_dir, pred_root, cases)
    object_summary, object_csv = write_object_outputs(object_data, paper_dir)
    object_panel = render_object_panel(object_data, paper_dir / "respan_paper_matched_holdout_panel.png", dataset_title)

    outputs = {
        "overlays": [str(path) for path in overlay_outputs],
        "metrics": [str(summary_csv), str(by_case_csv), str(summary_json), str(metric_plot)],
        "object_metrics": [str(object_summary), str(object_csv), str(object_panel)],
    }
    manifest = evaluation_dir / "respan_style_outputs_manifest.json"
    manifest.write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")
    outputs["manifest"] = [str(manifest)]
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["invivo", "confocal", "all"], default="all")
    args = parser.parse_args()

    datasets = ["invivo", "confocal"] if args.dataset == "all" else [args.dataset]
    all_outputs = {dataset: render_dataset(dataset) for dataset in datasets}
    print(json.dumps(all_outputs, indent=2))


if __name__ == "__main__":
    main()
