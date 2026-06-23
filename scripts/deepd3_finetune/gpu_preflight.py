#!/usr/bin/env python3
"""Verify GPU visibility and run a tiny TensorFlow convolution."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    report = {}
    nvidia = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    report["nvidia_smi_returncode"] = nvidia.returncode
    report["nvidia_smi_stdout"] = nvidia.stdout
    report["nvidia_smi_stderr"] = nvidia.stderr

    import tensorflow as tf

    report["tensorflow_version"] = tf.__version__
    report["tensorflow_built_with_cuda"] = bool(tf.test.is_built_with_cuda())
    gpus = tf.config.list_physical_devices("GPU")
    report["gpus"] = [gpu.name for gpu in gpus]

    if not gpus and not args.allow_cpu:
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        print("No TensorFlow GPU detected.", file=sys.stderr)
        raise SystemExit(2)

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    device = "/GPU:0" if gpus else "/CPU:0"
    with tf.device(device):
        x = tf.random.uniform((1, 32, 32, 1))
        y = tf.keras.layers.Conv2D(4, 3, padding="same")(x)
        report["test_device"] = device
        report["test_tensor_shape"] = list(y.shape)
        report["test_tensor_mean"] = float(tf.reduce_mean(y).numpy())

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
