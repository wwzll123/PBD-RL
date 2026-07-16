#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/inference.yaml}"
if [ "$#" -gt 0 ]; then
  shift
fi

if command -v conda >/dev/null 2>&1; then
  conda activate "${CONDA_ENV_NAME:-pxdesign}" 2>/dev/null || true
fi

python scripts/run_pxdesign_infer.py --config "${CONFIG}" "$@"
