#!/usr/bin/env python3
"""Download DeepD3 model zoo weights into the local model cache."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from common import MODEL_ZOO_URLS, PRETRAINED_MODEL_DIR, ensure_dirs


def download_model(name: str, out_dir: Path, force: bool) -> Path:
    if name not in MODEL_ZOO_URLS:
        known = ", ".join(sorted(MODEL_ZOO_URLS))
        raise KeyError(f"Unknown model {name!r}. Known models: {known}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    if out_path.exists() and not force:
        print(f"Using existing {out_path}")
        return out_path

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    url = MODEL_ZOO_URLS[name]
    print(f"Downloading {url} -> {out_path}")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as f:
        shutil.copyfileobj(response, f)
    tmp_path.replace(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["DeepD3_8F.h5", "DeepD3_16F.h5"],
        help="Model zoo filenames to cache.",
    )
    parser.add_argument("--out-dir", type=Path, default=PRETRAINED_MODEL_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    for model_name in args.models:
        path = download_model(model_name, args.out_dir, args.force)
        print(path)


if __name__ == "__main__":
    main()
