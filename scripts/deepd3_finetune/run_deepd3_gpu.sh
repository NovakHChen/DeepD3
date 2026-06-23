#!/usr/bin/env bash
set -euo pipefail

conda run -n deepd3 bash -lc '
set -euo pipefail
NVIDIA_ROOT="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
if [ -d "$NVIDIA_ROOT" ]; then
  NVIDIA_LIBS=$(find "$NVIDIA_ROOT" -type d -name lib | paste -sd: -)
  export LD_LIBRARY_PATH="$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}"
fi
exec python "$@"
' _ "$@"
