#!/usr/bin/env bash

set -e

GPU_ID=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# 基础配置
BASE_CONFIG="configs/dccrn_base.yaml"

# 实验名称及输出目录
EXP_NAME="dccrn_snr_m10_10_lr=1e-4_si_snr=0.9,smse=0.1"
OUTPUT_DIR="logs/${EXP_NAME}"

# 数据列表
TRAIN_CLEAN_LIST="data/ShipsEar-12class_W5H1/train_16k_signal.list"
TRAIN_NOISE_LIST="data/ShipsEar-12class_W5H1/train_16k_noise.list"
VALID_PAIR_LIST="data/ShipsEar-12class_W5H1/valid/list/valid_snr_-5dB.list"
TEST_PAIR_LIST="data/ShipsEar-12class_W5H1/test/list/valid_snr_-5dB.list"

# 数据参数
SAMPLE_RATE=16000
SEGMENT_SECONDS=5.0
SNR_MIN=-10
SNR_MAX=10
NUM_WORKERS=2

# 训练参数
SEED=42
BATCH_SIZE=32
EPOCHS=50
LEARNING_RATE=1e-4

# 留空表示从头训练
RESUME=""
# RESUME="runs/dccrn_snr_m5_20/checkpoints/last.pt"


# ==================================================
# 一般不需要修改下面内容
# ==================================================

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJECT_DIR}"

GENERATED_CONFIG="configs/generated/${EXP_NAME}.yaml"

mkdir -p "configs/generated"
mkdir -p "${OUTPUT_DIR}"

python - <<PY
import yaml

base_config = "${BASE_CONFIG}"
output_config = "${GENERATED_CONFIG}"

with open(base_config, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 实验设置
config["experiment"]["name"] = "${EXP_NAME}"
config["experiment"]["output_dir"] = "${OUTPUT_DIR}"
config["experiment"]["seed"] = ${SEED}

# 数据设置
config["data"]["train_clean_list"] = "${TRAIN_CLEAN_LIST}"
config["data"]["train_noise_list"] = "${TRAIN_NOISE_LIST}"
config["data"]["valid_pair_list"] = "${VALID_PAIR_LIST}"
config["data"]["test_pair_list"] = "${TEST_PAIR_LIST}"

config["data"]["sample_rate"] = ${SAMPLE_RATE}
config["data"]["segment_seconds"] = ${SEGMENT_SECONDS}
config["data"]["train_snr_min"] = ${SNR_MIN}
config["data"]["train_snr_max"] = ${SNR_MAX}
config["data"]["num_workers"] = ${NUM_WORKERS}

# 训练设置
config["training"]["batch_size"] = ${BATCH_SIZE}
config["training"]["epochs"] = ${EPOCHS}
config["training"]["learning_rate"] = ${LEARNING_RATE}

with open(output_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(
        config,
        f,
        allow_unicode=True,
        sort_keys=False,
    )

print("生成配置文件：", output_config)
PY


echo "=========================================="
echo "实验名称：${EXP_NAME}"
echo "GPU：${GPU_ID}"
echo "训练SNR：${SNR_MIN}～${SNR_MAX} dB"
echo "Batch size：${BATCH_SIZE}"
echo "Epochs：${EPOCHS}"
echo "学习率：${LEARNING_RATE}"
echo "配置文件：${GENERATED_CONFIG}"
echo "输出目录：${OUTPUT_DIR}"
echo "=========================================="


if [ -n "${RESUME}" ]; then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python train.py \
        --config "${GENERATED_CONFIG}" \
        --resume "${RESUME}" \
        2>&1 | tee "${OUTPUT_DIR}/console.log"
else
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python train.py \
        --config "${GENERATED_CONFIG}" \
        2>&1 | tee "${OUTPUT_DIR}/console.log"
fi