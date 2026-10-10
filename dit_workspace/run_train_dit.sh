#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Conditional DiT 训练脚本
# 只需要修改“参数配置”区域
# ============================================================

# ===================== 参数配置 =====================

PROJECT_DIR="/data/huayifan/water_denoise/dit_workspace"

# ---------------- GPU ----------------
# 单卡：
GPU_IDS="0"
NUM_GPUS="1"

# 多卡示例：
# GPU_IDS="0,1,2,3"
# NUM_GPUS="4"

# CPU：
# GPU_IDS=""
# NUM_GPUS="0"

# ---------------- 基础配置 ----------------
# Score + complex STFT
# BASE_CONFIG="configs/dit_score_complex_small.yaml"
BASE_CONFIG="configs/dit_score_complex_small_frame=256.yaml"

# 其他配置：
# BASE_CONFIG="configs/dit_score_magnitude_small.yaml"
# BASE_CONFIG="configs/dit_ddpm_complex_small.yaml"
# BASE_CONFIG="configs/dit_flow_complex_small.yaml"

# ---------------- 实验名称 / 保存目录 ----------------
EXP_NAME="dit_score_complex_frame=256,seg=2s_onthefly"

# 可以随时修改整个实验保存根目录
RUN_ROOT="${PROJECT_DIR}/runs"

# 也可以直接写成绝对路径，例如：
# RUN_ROOT="/data/huayifan/experiments/dit"

RUN_DIR="${RUN_ROOT}/${EXP_NAME}"

# ---------------- 数据模式 ----------------
# paired:
#   TRAIN_MANIFEST 每行 noisy<TAB>clean
#
# on_the_fly:
#   TRAIN_MANIFEST 为 clean 单列 list
#   NOISE_MANIFEST 为 noise 单列 list
#
DATA_MODE="on_the_fly"

# ---------------- 训练 / 验证 / 默认测试 list ----------------

# on_the_fly 模式：
#   TRAIN_MANIFEST 为 clean 单列 list
#   NOISE_MANIFEST 为 noise 单列 list
#   VALID/TEST 继续使用固定 noisy<TAB>clean 配对 list，便于稳定验证/测试
TRAIN_MANIFEST="/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/train_16k_signal.list"
NOISE_MANIFEST="/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/train_16k_noise.list"
VALID_MANIFEST="/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/valid/list/valid_snr_-5dB.list"
TEST_MANIFEST="/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/test/list/snr_-5dB.list"

# ---------------- 在线混噪参数 ----------------
# paired 模式下这两个参数不会改变已有 noisy/clean 数据
SNR_MIN="-15"
SNR_MAX="0"

# ---------------- 训练参数 ----------------
SEED="42"
EPOCHS="100"
BATCH_SIZE="32"
LEARNING_RATE="0.0001"
NUM_WORKERS="4"
AMP="False"

# ---------------- 验证参数 ----------------
EVAL_INTERVAL="5"
MAX_SAMPLING_BATCHES="2"
VALIDATION_SEED="1234"

SAVE_EVERY="10"

# ---------------- 断点续训 ----------------
# 留空表示从头训练
RESUME_CHECKPOINT=""

# 例如：
# RESUME_CHECKPOINT="${RUN_DIR}/checkpoints/last.pt"

# ============================================================


# ===================== 自动生成路径 =====================

RUNTIME_CONFIG="${RUN_DIR}/runtime_config.yaml"
CONSOLE_LOG="${RUN_DIR}/console.log"

cd "${PROJECT_DIR}"

mkdir -p "${RUN_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONUNBUFFERED=1


# ============================================================
# 文件检查
# ============================================================

if [[ "${DATA_MODE}" != "on_the_fly" ]]; then
    echo "[ERROR] this script is configured for on_the_fly training; DATA_MODE=${DATA_MODE}" >&2
    exit 1
fi

if [[ -z "${NOISE_MANIFEST}" ]]; then
    echo "[ERROR] on_the_fly training requires a non-empty NOISE_MANIFEST" >&2
    exit 1
fi

for f in \
    "${BASE_CONFIG}" \
    "${TRAIN_MANIFEST}" \
    "${NOISE_MANIFEST}" \
    "${VALID_MANIFEST}" \
    "${TEST_MANIFEST}"
do
    if [[ ! -f "${f}" ]]; then
        echo "[ERROR] file not found: ${f}" >&2
        exit 1
    fi
done

if [[ -n "${RESUME_CHECKPOINT}" && ! -f "${RESUME_CHECKPOINT}" ]]; then
    echo "[ERROR] resume checkpoint not found: ${RESUME_CHECKPOINT}" >&2
    exit 1
fi


# ============================================================
# 生成训练 runtime_config.yaml
# ============================================================

python - \
    "${BASE_CONFIG}" \
    "${RUNTIME_CONFIG}" \
    "${EXP_NAME}" \
    "${RUN_DIR}" \
    "${DATA_MODE}" \
    "${TRAIN_MANIFEST}" \
    "${VALID_MANIFEST}" \
    "${TEST_MANIFEST}" \
    "${NOISE_MANIFEST}" \
    "${SNR_MIN}" \
    "${SNR_MAX}" \
    "${SEED}" \
    "${EPOCHS}" \
    "${BATCH_SIZE}" \
    "${LEARNING_RATE}" \
    "${NUM_WORKERS}" \
    "${AMP}" \
    "${EVAL_INTERVAL}" \
    "${MAX_SAMPLING_BATCHES}" \
    "${VALIDATION_SEED}" \
    "${SAVE_EVERY}" <<'PY'

import sys
from pathlib import Path
import yaml

(
    base_config,
    runtime_config,
    exp_name,
    run_dir,
    data_mode,
    train_manifest,
    valid_manifest,
    test_manifest,
    noise_manifest,
    snr_min,
    snr_max,
    seed,
    epochs,
    batch_size,
    learning_rate,
    num_workers,
    amp,
    eval_interval,
    max_sampling_batches,
    validation_seed,
    save_every,
) = sys.argv[1:]

with open(base_config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

if data_mode != "on_the_fly":
    raise ValueError(f"Expected data_mode=on_the_fly, got {data_mode!r}")
if not noise_manifest:
    raise ValueError("on_the_fly training requires noise_manifest")

# ------------------------------------------------------------
# Experiment
# ------------------------------------------------------------

cfg["experiment"]["name"] = exp_name
cfg["experiment"]["output_dir"] = str(Path(run_dir).resolve())
cfg["experiment"]["seed"] = int(seed)

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

cfg["data"]["data_mode"] = data_mode

cfg["data"]["train_manifest"] = str(Path(train_manifest).resolve())
cfg["data"]["valid_manifest"] = str(Path(valid_manifest).resolve())
cfg["data"]["test_manifest"] = str(Path(test_manifest).resolve())

if noise_manifest:
    cfg["data"]["noise_manifest"] = str(Path(noise_manifest).resolve())
else:
    cfg["data"]["noise_manifest"] = None

cfg["data"]["snr_min"] = float(snr_min)
cfg["data"]["snr_max"] = float(snr_max)
cfg["data"]["num_workers"] = int(num_workers)

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

cfg["training"]["epochs"] = int(epochs)
cfg["training"]["batch_size"] = int(batch_size)
cfg["training"]["learning_rate"] = float(learning_rate)
cfg["training"]["amp"] = amp.lower() == "true"
cfg["training"]["save_every"] = int(save_every)

# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

cfg.setdefault("validation", {})

cfg["validation"]["eval_interval"] = int(eval_interval)
cfg["validation"]["max_sampling_batches"] = int(max_sampling_batches)
cfg["validation"]["seed"] = int(validation_seed)

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------

cfg.setdefault("logging", {})
cfg["logging"]["tensorboard"] = True
cfg["logging"]["jsonl"] = True

# ------------------------------------------------------------
# Save runtime config
# ------------------------------------------------------------

runtime_path = Path(runtime_config)
runtime_path.parent.mkdir(parents=True, exist_ok=True)

with runtime_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(
        cfg,
        f,
        allow_unicode=True,
        sort_keys=False,
    )

print("Runtime config generated:")
print(runtime_path)

print("Representation :", cfg["representation"]["type"])
print("Process        :", cfg["process"]["type"])
print("Sampler        :", cfg["sampler"]["name"])
print("Patch size     :", cfg["model"]["patch_size"])

PY


# ============================================================
# 打印训练配置
# ============================================================

echo
echo "============================================================"
echo "Conditional DiT Training"
echo "============================================================"
echo "Project dir            : ${PROJECT_DIR}"
echo "Visible GPU            : ${GPU_IDS:-CPU}"
echo "Number of GPUs         : ${NUM_GPUS}"
echo
echo "Base config            : ${BASE_CONFIG}"
echo "Experiment             : ${EXP_NAME}"
echo "Run directory          : ${RUN_DIR}"
echo
echo "Data mode              : ${DATA_MODE}"
echo "Train manifest         : ${TRAIN_MANIFEST}"
echo "Valid manifest         : ${VALID_MANIFEST}"
echo "Test manifest          : ${TEST_MANIFEST}"
echo "Noise manifest         : ${NOISE_MANIFEST:-None}"
echo "SNR                    : ${SNR_MIN} ~ ${SNR_MAX} dB"
echo
echo "Epochs                 : ${EPOCHS}"
echo "Batch size             : ${BATCH_SIZE}"
echo "Learning rate          : ${LEARNING_RATE}"
echo "Workers                : ${NUM_WORKERS}"
echo
echo "Eval interval          : ${EVAL_INTERVAL}"
echo "Sampling valid batches : ${MAX_SAMPLING_BATCHES}"
echo "Validation seed        : ${VALIDATION_SEED}"
echo
echo "Runtime config         : ${RUNTIME_CONFIG}"
echo "Resume                 : ${RESUME_CHECKPOINT:-None}"
echo "Console log            : ${CONSOLE_LOG}"
echo "============================================================"
echo


# ============================================================
# 开始训练
# ============================================================

CMD=(
    python -u train.py
    --config "${RUNTIME_CONFIG}"
    --num-gpus "${NUM_GPUS}"
)

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    CMD+=(--resume "${RESUME_CHECKPOINT}")
fi

"${CMD[@]}" 2>&1 | tee "${CONSOLE_LOG}"


# ============================================================
# 输出结果
# ============================================================

echo
echo "============================================================"
echo "Training finished"
echo "============================================================"
echo "Best checkpoint : ${RUN_DIR}/checkpoints/best.pt"
echo "Last checkpoint : ${RUN_DIR}/checkpoints/last.pt"
echo "Runtime config  : ${RUN_DIR}/runtime_config.yaml"
echo "Metrics JSONL   : ${RUN_DIR}/metrics.jsonl"
echo "TensorBoard     : ${RUN_DIR}/tensorboard"
echo "Console log     : ${CONSOLE_LOG}"
echo "============================================================"