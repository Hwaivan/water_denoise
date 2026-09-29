#!/usr/bin/env bash
set -euo pipefail
# 参数配置区
GPU_ID="0"
EXP_NAME="dit_score_complex_small"
CONFIG_FILE="configs/dit_score_complex_small.yaml"
CHECKPOINT_FILE="runs/${EXP_NAME}/checkpoints/best.pt"
TEST_PAIR_LIST="data/test_pairs.txt"
TEST_NAME="test"
TEST_BATCH_SIZE=1
SAMPLER_SEED=1234
PROCESS_TYPE="score"       # Must agree with the trained checkpoint.
SAMPLER="pc"               # score: pc; ddpm: ddpm; flow: euler or heun
NUM_STEPS=30               # DDPM must equal its training schedule length.
DEVICE="cuda"              # cpu is also supported.
PYTHON="python"

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
RUNTIME_CONFIG="runs/${EXP_NAME}/runtime_config.yaml"
if [[ -f "$RUNTIME_CONFIG" ]]; then CONFIG_FILE="$RUNTIME_CONFIG"; fi
"$PYTHON" evaluate.py --config "$CONFIG_FILE" --checkpoint "$CHECKPOINT_FILE" \
  --test-manifest "$TEST_PAIR_LIST" --output-dir "runs/${EXP_NAME}/evaluation/${TEST_NAME}" \
  --batch-size "$TEST_BATCH_SIZE" --seed "$SAMPLER_SEED" --process "$PROCESS_TYPE" \
  --sampler "$SAMPLER" --num-steps "$NUM_STEPS" --device "$DEVICE"
