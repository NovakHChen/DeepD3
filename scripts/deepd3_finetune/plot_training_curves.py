#!/usr/bin/env python3
"""Plot DeepD3 fine-tuning history curves saved during training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FINETUNED_MODEL_DIR


def load_history(dataset: str) -> pd.DataFrame:
    rows = []
    model_dir = FINETUNED_MODEL_DIR / dataset
    for stage_name, offset, path in [
        ("stage1", 0, model_dir / "stage1_history.csv"),
        ("stage2", 5, model_dir / "stage2_history.csv"),
    ]:
        df = pd.read_csv(path)
        df["stage"] = stage_name
        df["global_epoch"] = df["epoch"].astype(int) + 1 + offset
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def plot_dataset(dataset: str) -> Path:
    df = load_history(dataset)
    out_path = FINETUNED_MODEL_DIR / dataset / "training_curves.png"

    panels = [
        ("loss", "val_loss", "Total Loss"),
        ("dendrites_loss", "val_dendrites_loss", "Dendrite Loss"),
        ("spines_loss", "val_spines_loss", "Spine Loss"),
        ("spines_iou_score", "val_spines_iou_score", "Spine IoU"),
    ]
    title = "In-vivo" if dataset == "invivo" else "Confocal"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (train_key, val_key, label) in zip(axes.ravel(), panels):
        ax.plot(df["global_epoch"], df[train_key], marker="o", linewidth=1.5, label="train")
        ax.plot(df["global_epoch"], df[val_key], marker="o", linewidth=1.5, label="validation")
        ax.axvline(5.5, color="#888888", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f"{title} DeepD3 Fine-Tuning Curves", fontsize=15, fontweight="bold")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["invivo", "confocal", "all"], default="all")
    args = parser.parse_args()

    datasets = ["invivo", "confocal"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        print(plot_dataset(dataset))


if __name__ == "__main__":
    main()
