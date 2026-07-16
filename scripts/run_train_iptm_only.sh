#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/train_iptm_only.yaml}"
shift || true

EXTRA_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --max_steps)
      EXTRA_ARGS+=("training.max_steps=$2")
      shift 2
      ;;
    --nproc_per_node)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if command -v conda >/dev/null 2>&1; then
  conda activate "${CONDA_ENV_NAME:-pxdesign}" 2>/dev/null || true
fi

CONFIG_DIR="$(dirname "$CONFIG_PATH")"
CONFIG_BASE="$(basename "$CONFIG_PATH")"
CONFIG_NAME="${CONFIG_BASE%.*}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

if [ "$NPROC_PER_NODE" -gt 1 ]; then
  torchrun --nproc_per_node "$NPROC_PER_NODE" train.py --config-path "$CONFIG_DIR" --config-name "$CONFIG_NAME" "${EXTRA_ARGS[@]}"
else
  python train.py --config-path "$CONFIG_DIR" --config-name "$CONFIG_NAME" "${EXTRA_ARGS[@]}"
fi
