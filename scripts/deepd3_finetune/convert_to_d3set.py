#!/usr/bin/env python3
"""Convert copied nnU-Net-style TIFF data into DeepD3 d3set files."""

from __future__ import annotations

import argparse
from pathlib import Path

import flammkuchen as fl
import pandas as pd
import tifffile

from common import (
    DATASETS,
    D3SET_DIR,
    METADATA_DIR,
    collect_split_cases,
    ensure_dirs,
    image_path_for_case,
    label_path_for_case,
    load_invivo_fallbacks,
    load_metadata,
    resolve_spacing,
    validate_case_pair,
    write_json,
)


def convert_dataset_split(
    dataset_key: str,
    split: str,
    metadata: pd.DataFrame,
    invivo_fallbacks: dict[str, dict[str, float]],
    out_dir: Path,
    overwrite: bool,
) -> dict:
    spec = DATASETS[dataset_key]
    dataset_dir = spec.target_dir
    cases = collect_split_cases(dataset_dir, split)
    out_path = out_dir / f"{dataset_key}_{split}.d3set"
    manifest_path = out_dir / f"{dataset_key}_{split}.manifest.json"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists; pass --overwrite to replace it")

    stacks = {}
    dendrites = {}
    spines = {}
    meta_rows = []
    manifest_cases = []

    for idx, case_id in enumerate(cases):
        image_path = image_path_for_case(dataset_dir, split, case_id)
        label_path = label_path_for_case(dataset_dir, split, case_id)
        validation = validate_case_pair(image_path, label_path)
        stack = tifffile.imread(image_path)
        label = tifffile.imread(label_path)

        spacing = resolve_spacing(case_id, spec, metadata, invivo_fallbacks)
        key = f"x{idx}"
        stacks[key] = stack
        dendrites[key] = label == 2
        spines[key] = label == 1

        depth, height, width = stack.shape
        meta_rows.append(
            {
                "Case": case_id,
                "Dataset": spec.target_name,
                "Split": split,
                "Width": int(width),
                "Height": int(height),
                "Depth": int(depth),
                "Resolution_XY": float(spacing["resolution_xy"]),
                "Resolution_Z": float(spacing["resolution_z"]),
                "Generated_from": str(image_path),
                "Label_from": str(label_path),
                "Metadata_name": spacing["metadata_name"],
                "Metadata_match_type": spacing["match_type"],
            }
        )
        manifest_cases.append(
            {
                "case_id": case_id,
                "d3set_key": key,
                "image": str(image_path),
                "label": str(label_path),
                "validation": validation,
                "spacing": spacing,
            }
        )

    meta = pd.DataFrame(meta_rows)
    fl.save(
        out_path,
        {
            "data": {
                "stacks": stacks,
                "dendrites": dendrites,
                "spines": spines,
            },
            "meta": meta,
        },
        compression="blosc",
    )

    manifest = {
        "dataset_key": dataset_key,
        "dataset": spec.target_name,
        "split": split,
        "case_count": len(cases),
        "output": str(out_path),
        "cases": manifest_cases,
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=METADATA_DIR / "inspire_metadata.xlsx")
    parser.add_argument(
        "--invivo-fallback",
        type=Path,
        default=METADATA_DIR / "invivo_missing_voxel_metadata_template.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=D3SET_DIR)
    parser.add_argument("--dataset", choices=sorted(DATASETS), action="append")
    parser.add_argument("--split", choices=["train", "heldout"], action="append")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.metadata)
    invivo_fallbacks = load_invivo_fallbacks(args.invivo_fallback)
    datasets = args.dataset or ["invivo", "confocal"]
    splits = args.split or ["train", "heldout"]

    manifests = []
    for dataset_key in datasets:
        for split in splits:
            manifest = convert_dataset_split(
                dataset_key,
                split,
                metadata,
                invivo_fallbacks,
                args.out_dir,
                args.overwrite,
            )
            manifests.append(manifest)
            print(f"Wrote {manifest['output']} ({manifest['case_count']} cases)")

    write_json(args.out_dir / "conversion_manifest.json", manifests)
    print(f"Wrote {args.out_dir / 'conversion_manifest.json'}")


if __name__ == "__main__":
    main()
