#!/usr/bin/env python3
"""Shared helpers for the local DeepD3 fine-tuning workflow."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = Path("/home/nochen/code/respan_finetune/nnUNet_raw")
DEFAULT_INSPIRE_METADATA = Path("/home/nochen/code/inspire/data/metadata.xlsx")
DEFAULT_INVIVO_FALLBACK_CSV = (
    Path("/home/nochen/code/respan_finetune")
    / "annotations/invivo_kg/Dataset003_invivo/missing_voxel_metadata_template.csv"
)

DATA_ROOT = REPO_ROOT / "data"
METADATA_DIR = DATA_ROOT / "metadata"
ARTIFACT_ROOT = REPO_ROOT / "artifacts"
D3SET_DIR = ARTIFACT_ROOT / "deepd3_datasets"
MODEL_ROOT = REPO_ROOT / "models"
PRETRAINED_MODEL_DIR = MODEL_ROOT / "pretrained"
FINETUNED_MODEL_DIR = MODEL_ROOT / "respan_finetuned"

LABEL_VALUES = {"background": 0, "spine": 1, "dendrite": 2}
ALLOWED_LABEL_VALUES = set(LABEL_VALUES.values())

MODEL_ZOO_URLS = {
    "DeepD3_8F.h5": "https://deepd3.forschung.fau.de/models/DeepD3_8F.h5",
    "DeepD3_16F.h5": "https://deepd3.forschung.fau.de/models/DeepD3_16F.h5",
    "DeepD3_32F.h5": "https://deepd3.forschung.fau.de/models/DeepD3_32F.h5",
    "DeepD3_8F_94nm.h5": "https://deepd3.forschung.fau.de/models/DeepD3_8F_94nm.h5",
    "DeepD3_16F_94nm.h5": "https://deepd3.forschung.fau.de/models/DeepD3_16F_94nm.h5",
    "DeepD3_32F_94nm.h5": "https://deepd3.forschung.fau.de/models/DeepD3_32F_94nm.h5",
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    source_name: str
    target_name: str
    xy_column: str
    z_column: str

    @property
    def target_dir(self) -> Path:
        return DATA_ROOT / self.target_name


DATASETS: dict[str, DatasetSpec] = {
    "invivo": DatasetSpec(
        key="invivo",
        source_name="Dataset003_invivo",
        target_name="Dataset003_invivo",
        xy_column="invivo_voxel",
        z_column="invivo_zstep",
    ),
    "confocal": DatasetSpec(
        key="confocal",
        source_name="Dataset002_kgconfocal",
        target_name="Dataset002_kgconfocal",
        xy_column="confocal_voxel",
        z_column="confocal_zstep",
    ),
}

# INSPIRE metadata uses bd15/bd16 while the TIFF cases use d15/d16.
METADATA_ALIASES = {
    "kg215d15": "kg215bd15",
    "kg215d16": "kg215bd16",
}


def ensure_dirs() -> None:
    for path in [
        DATA_ROOT,
        METADATA_DIR,
        D3SET_DIR,
        PRETRAINED_MODEL_DIR,
        FINETUNED_MODEL_DIR / "invivo",
        FINETUNED_MODEL_DIR / "confocal",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def is_real_tiff(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in {".tif", ".tiff"}
        and not path.name.startswith("._")
        and path.name not in {".DS_Store", "Thumbs.db"}
    )


def case_from_image(path: Path) -> str:
    stem = path.stem
    return stem[:-5] if stem.endswith("_0000") else stem


def case_from_label(path: Path) -> str:
    stem = path.stem
    return stem[:-5] if stem.endswith("_0000") else stem


def normalize_metadata_key(name: str) -> str:
    stem = Path(str(name)).stem
    stem = re.sub(r"_u\d+$", "", stem, flags=re.IGNORECASE)
    compact = re.sub(r"[^0-9a-z]", "", stem.lower())
    return re.sub(r"\d+", lambda match: str(int(match.group())), compact)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def collect_split_cases(dataset_dir: Path, split: str) -> list[str]:
    manifest_path = dataset_dir / "split_manifest.json"
    manifest = read_json(manifest_path)
    key = "training_cases" if split == "train" else "heldout_cases"
    return list(manifest[key])


def split_dirs(dataset_dir: Path, split: str) -> tuple[Path, Path]:
    suffix = "Tr" if split == "train" else "Ts"
    return dataset_dir / f"images{suffix}", dataset_dir / f"labels{suffix}"


def image_path_for_case(dataset_dir: Path, split: str, case_id: str) -> Path:
    image_dir, _ = split_dirs(dataset_dir, split)
    return image_dir / f"{case_id}_0000.tif"


def label_path_for_case(dataset_dir: Path, split: str, case_id: str) -> Path:
    _, label_dir = split_dirs(dataset_dir, split)
    return label_dir / f"{case_id}.tif"


def validate_case_pair(image_path: Path, label_path: Path) -> dict[str, Any]:
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label: {label_path}")

    image = tifffile.imread(image_path)
    label = tifffile.imread(label_path)
    if image.shape != label.shape:
        raise ValueError(
            f"Shape mismatch for {image_path.stem}: image {image.shape}, label {label.shape}"
        )

    values, counts = np.unique(label, return_counts=True)
    value_counts = {int(v): int(c) for v, c in zip(values, counts)}
    unexpected = sorted(set(value_counts) - ALLOWED_LABEL_VALUES)
    if unexpected:
        raise ValueError(f"Unexpected label values for {label_path}: {unexpected}")

    return {
        "image_shape": list(image.shape),
        "image_dtype": str(image.dtype),
        "label_dtype": str(label.dtype),
        "label_value_counts": value_counts,
    }


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    df = pd.read_excel(metadata_path)
    required = {"name", "invivo_voxel", "invivo_zstep", "confocal_voxel", "confocal_zstep"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Metadata workbook is missing columns: {missing}")
    out = df.copy()
    out["_normalized_name"] = out["name"].map(normalize_metadata_key)
    return out


def load_invivo_fallbacks(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None or not path.exists():
        return {
            "kg193tuft2": {
                "invivo_voxel": 0.0453210389,
                "invivo_zstep": 1.0,
            }
        }

    fallbacks: dict[str, dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("case_id"):
                continue
            fallbacks[normalize_metadata_key(row["case_id"])] = {
                "invivo_voxel": float(row["invivo_voxel_um_xy"]),
                "invivo_zstep": float(row["invivo_zstep_um"]),
            }
    return fallbacks


def resolve_spacing(
    case_id: str,
    spec: DatasetSpec,
    metadata: pd.DataFrame,
    invivo_fallbacks: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    key = normalize_metadata_key(case_id)
    lookup_key = METADATA_ALIASES.get(key, key)
    match = metadata[metadata["_normalized_name"] == lookup_key]
    if not match.empty:
        row = match.iloc[0]
        return {
            "case_id": case_id,
            "resolution_xy": float(row[spec.xy_column]),
            "resolution_z": float(row[spec.z_column]),
            "metadata_name": str(row["name"]),
            "match_type": "metadata_exact" if lookup_key == key else "metadata_alias",
        }

    if spec.key == "confocal":
        xy_values = metadata[spec.xy_column].dropna().unique()
        z_values = metadata[spec.z_column].dropna().unique()
        if len(xy_values) == 1 and len(z_values) == 1:
            return {
                "case_id": case_id,
                "resolution_xy": float(xy_values[0]),
                "resolution_z": float(z_values[0]),
                "metadata_name": None,
                "match_type": "global_confocal_default",
            }

    if spec.key == "invivo" and invivo_fallbacks:
        fallback = invivo_fallbacks.get(key)
        if fallback:
            return {
                "case_id": case_id,
                "resolution_xy": float(fallback["invivo_voxel"]),
                "resolution_z": float(fallback["invivo_zstep"]),
                "metadata_name": None,
                "match_type": "invivo_fallback_csv",
            }

    raise KeyError(f"No spacing metadata found for {spec.key} case {case_id}")


def dice_iou(pred_mask: np.ndarray, ref_mask: np.ndarray) -> dict[str, Any]:
    pred = pred_mask.astype(bool)
    ref = ref_mask.astype(bool)
    tp = int(np.logical_and(pred, ref).sum())
    fp = int(np.logical_and(pred, ~ref).sum())
    fn = int(np.logical_and(~pred, ref).sum())
    denom_dice = (2 * tp) + fp + fn
    denom_iou = tp + fp + fn
    return {
        "Dice": float((2 * tp) / denom_dice) if denom_dice else 1.0,
        "IoU": float(tp / denom_iou) if denom_iou else 1.0,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "n_pred": int(pred.sum()),
        "n_ref": int(ref.sum()),
    }
