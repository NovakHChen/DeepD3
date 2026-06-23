#!/usr/bin/env python3
"""Copy RESPAN raw TIFF datasets into the DeepD3 repo and validate them."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import (
    DATASETS,
    DATA_ROOT,
    DEFAULT_INSPIRE_METADATA,
    DEFAULT_INVIVO_FALLBACK_CSV,
    DEFAULT_SOURCE_ROOT,
    METADATA_DIR,
    collect_split_cases,
    ensure_dirs,
    image_path_for_case,
    label_path_for_case,
    read_json,
    validate_case_pair,
    write_json,
)


def copy_file(src: Path, dst: Path, overwrite: bool) -> str:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return "skipped_existing"
    shutil.copy2(src, dst)
    return "copied"


def copy_dataset(source_root: Path, dataset_key: str, overwrite: bool) -> dict:
    spec = DATASETS[dataset_key]
    source_dir = source_root / spec.source_name
    target_dir = DATA_ROOT / spec.target_name
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    report = {
        "dataset": spec.target_name,
        "source": str(source_dir),
        "target": str(target_dir),
        "files": [],
        "splits": {},
    }

    for top_file in ["dataset.json", "split_manifest.json"]:
        status = copy_file(source_dir / top_file, target_dir / top_file, overwrite)
        report["files"].append({"source": str(source_dir / top_file), "status": status})

    for subdir in ["imagesTr", "labelsTr", "imagesTs", "labelsTs"]:
        src_dir = source_dir / subdir
        dst_dir = target_dir / subdir
        dst_dir.mkdir(parents=True, exist_ok=True)
        if not src_dir.exists():
            continue
        for src in sorted(src_dir.iterdir()):
            if src.name.startswith("._") or src.name in {".DS_Store", "Thumbs.db"}:
                continue
            if src.suffix.lower() not in {".tif", ".tiff", ".json"}:
                continue
            status = copy_file(src, dst_dir / src.name, overwrite)
            report["files"].append({"source": str(src), "target": str(dst_dir / src.name), "status": status})

    for split, expected_count in [("train", 19), ("heldout", 5)]:
        cases = collect_split_cases(target_dir, split)
        if len(cases) != expected_count:
            raise ValueError(f"{spec.target_name} {split} expected {expected_count} cases, found {len(cases)}")
        split_report = {"case_count": len(cases), "cases": {}}
        for case_id in cases:
            image_path = image_path_for_case(target_dir, split, case_id)
            label_path = label_path_for_case(target_dir, split, case_id)
            split_report["cases"][case_id] = validate_case_pair(image_path, label_path)
        report["splits"][split] = split_report

    dataset_json = read_json(target_dir / "dataset.json")
    report["label_convention"] = dataset_json.get("labels", {})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--metadata-source", type=Path, default=DEFAULT_INSPIRE_METADATA)
    parser.add_argument("--invivo-fallback-source", type=Path, default=DEFAULT_INVIVO_FALLBACK_CSV)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ensure_dirs()

    metadata_copy = METADATA_DIR / "inspire_metadata.xlsx"
    copy_file(args.metadata_source, metadata_copy, args.overwrite)

    if args.invivo_fallback_source.exists():
        copy_file(
            args.invivo_fallback_source,
            METADATA_DIR / "invivo_missing_voxel_metadata_template.csv",
            args.overwrite,
        )

    report = {
        "metadata": {
            "source": str(args.metadata_source),
            "copy": str(metadata_copy),
        },
        "datasets": {},
    }

    for dataset_key in ["invivo", "confocal"]:
        report["datasets"][dataset_key] = copy_dataset(args.source_root, dataset_key, args.overwrite)

    out = DATA_ROOT / "copy_validation_manifest.json"
    write_json(out, report)
    print(f"Wrote {out}")
    for dataset_key, dataset_report in report["datasets"].items():
        train_n = dataset_report["splits"]["train"]["case_count"]
        heldout_n = dataset_report["splits"]["heldout"]["case_count"]
        print(f"{dataset_key}: {train_n} train, {heldout_n} heldout validated")


if __name__ == "__main__":
    main()
