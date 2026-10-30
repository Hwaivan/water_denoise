#!/usr/bin/env bash
set -euo pipefail

cd /data/huayifan/water_denoise/dit_workspace

echo "=========================================="
echo "Experiment 1: Patch-DiT crop256"
echo "=========================================="

BASE_CONFIG="configs/dit_score_complex_small_frame=256.yaml" \
EXP_NAME="dit_patch256_onthefly" \
BATCH_SIZE=16 \
GPU_IDS=0 \
NUM_GPUS=1 \
bash run_train_dit.sh


echo "=========================================="
echo "Experiment 2: Frame-DiT 1s"
echo "=========================================="

BASE_CONFIG="configs/dit_score_complex_frame_token_1s.yaml" \
EXP_NAME="dit_frame1d_1s_onthefly" \
BATCH_SIZE=32 \
GPU_IDS=0 \
NUM_GPUS=1 \
bash run_train_dit.sh


echo "=========================================="
echo "Experiment 3: Frame-DiT 5s"
echo "=========================================="

BASE_CONFIG="configs/dit_score_complex_frame_token_5s.yaml" \
EXP_NAME="dit_frame1d_5s_onthefly" \
BATCH_SIZE=16 \
GPU_IDS=0 \
NUM_GPUS=1 \
bash run_train_dit.sh


echo "=========================================="
echo "All experiments finished."
echo "=========================================="