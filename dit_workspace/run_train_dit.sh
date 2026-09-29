#!/usr/bin/env bash
set -euo pipefail
# 参数配置区 / run from any directory; data paths remain workspace-relative.
GPU_ID="0"
NUM_GPUS=1                 # 0: CPU, 1: single GPU, >1: native spawned DDP
CONFIG_FILE="configs/dit_score_complex_small.yaml"
RESUME=""
PYTHON="python"

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
args=(--config "$CONFIG_FILE" --num-gpus "$NUM_GPUS")
if [[ -n "$RESUME" ]]; then args+=(--resume "$RESUME"); fi
"$PYTHON" train.py "${args[@]}"
