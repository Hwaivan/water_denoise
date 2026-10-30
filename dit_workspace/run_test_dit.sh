#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Conditional DiT 测试脚本
# 只需要修改“参数配置”区域
# ============================================================

# ===================== 参数配置 =====================

PROJECT_DIR="/data/huayifan/water_denoise/dit_workspace"

# ---------------- GPU ----------------

GPU_ID="0"
DEVICE="cuda"

# CPU：
# DEVICE="cpu"

# ---------------- 训练实验 ----------------

# EXP_NAME="dit_score_complex_onthefly"
EXP_NAME="dit_score_complex_frame=256,seg=2s_onthefly"

RUN_ROOT="${PROJECT_DIR}/runs"
RUN_DIR="${RUN_ROOT}/${EXP_NAME}"

# 训练时自动保存的完整 runtime config
CONFIG_FILE="${RUN_DIR}/runtime_config.yaml"

# 默认使用 best checkpoint
CHECKPOINT_FILE="${RUN_DIR}/checkpoints/best.pt"

# 也可以测试 last.pt
# CHECKPOINT_FILE="${RUN_DIR}/checkpoints/last.pt"


# ---------------- 测试数据 ----------------

# 每行：
# noisy.wav<TAB>clean.wav

TEST_PAIR_LIST="/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/test/list/snr_0dB.list"

# 其他测试集：
# TEST_PAIR_LIST="/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/test/list/snr_0dB.list"
# TEST_PAIR_LIST="/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/test/list/snr_5dB.list"


# ---------------- 本次测试名称 ----------------

TEST_NAME="snr_0dB"

# 可以独立修改测试保存根目录
EVAL_ROOT="${RUN_DIR}/evaluation"

# 例如：
# EVAL_ROOT="/data/huayifan/evaluation/dit"

OUTPUT_DIR="${EVAL_ROOT}/${TEST_NAME}"

# 每次测试都保留自己的配置
TEST_RUNTIME_CONFIG="${OUTPUT_DIR}/test_runtime.yaml"


# ---------------- 测试参数 ----------------

TEST_BATCH_SIZE="1"
SAMPLER_SEED="1234"


# ---------------- sampler override ----------------
#
# 留空 = 完全使用训练 runtime config
#
# Score:
#   pc
#
# DDPM:
#   ddpm
#
# Flow:
#   euler
#   heun
#
SAMPLER_OVERRIDE=""

# 留空 = 使用训练配置中的 num_steps
NUM_STEPS_OVERRIDE=""

# Score 例：
# SAMPLER_OVERRIDE="pc"
# NUM_STEPS_OVERRIDE="30"

# Flow 例：
# SAMPLER_OVERRIDE="heun"
# NUM_STEPS_OVERRIDE="20"

# DDPM 当前必须完整执行训练 schedule，
# 因此通常不要随意减少 num_steps。


# ============================================================

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"

mkdir -p "${OUTPUT_DIR}"


# ============================================================
# 文件检查
# ============================================================

for f in \
    "${CONFIG_FILE}" \
    "${CHECKPOINT_FILE}" \
    "${TEST_PAIR_LIST}"
do
    if [[ ! -f "${f}" ]]; then
        echo "[ERROR] file not found: ${f}" >&2
        exit 1
    fi
done


# ============================================================
# 生成测试专用 test_runtime.yaml
# ============================================================

python - \
    "${CONFIG_FILE}" \
    "${TEST_RUNTIME_CONFIG}" \
    "${TEST_PAIR_LIST}" \
    "${TEST_BATCH_SIZE}" \
    "${SAMPLER_SEED}" \
    "${SAMPLER_OVERRIDE}" \
    "${NUM_STEPS_OVERRIDE}" <<'PY'

import sys
from pathlib import Path
import yaml

(
    config_file,
    runtime_file,
    test_manifest,
    batch_size,
    seed,
    sampler_override,
    num_steps_override,
) = sys.argv[1:]

with open(config_file, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# ------------------------------------------------------------
# Test dataset
# ------------------------------------------------------------

cfg["data"]["test_manifest"] = str(Path(test_manifest).resolve())

# 当前 dataloader 使用 training.batch_size
cfg["training"]["batch_size"] = int(batch_size)

# ------------------------------------------------------------
# Seed
# ------------------------------------------------------------

cfg.setdefault("validation", {})
cfg["validation"]["seed"] = int(seed)

cfg.setdefault("inference", {})
cfg["inference"]["seed"] = int(seed)

# ------------------------------------------------------------
# Optional sampler override
# ------------------------------------------------------------

process = cfg["process"]["type"]

# Predictor-only ablation
if process == "score":
    cfg["sampler"]["corrector"] = "none"
    cfg["sampler"]["corrector_steps"] = 0

if sampler_override:
    allowed = {
        "score": {"pc"},
        "ddpm": {"ddpm"},
        "flow": {"euler", "heun"},
    }

    if sampler_override not in allowed[process]:
        raise ValueError(
            f"Sampler '{sampler_override}' is invalid for "
            f"process '{process}'. "
            f"Allowed: {sorted(allowed[process])}"
        )

    cfg["sampler"]["name"] = sampler_override

if num_steps_override:
    steps = int(num_steps_override)

    # 当前 DDPM 实现要求完整 ancestral schedule
    if process == "ddpm":
        train_steps = int(cfg["process"]["ddpm"]["num_steps"])

        if steps != train_steps:
            raise ValueError(
                f"Current DDPM sampler requires num_steps={train_steps}, "
                f"but got {steps}. "
                "Reduced-step DDIM is not implemented."
            )

    cfg["sampler"]["num_steps"] = steps

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

runtime_path = Path(runtime_file)
runtime_path.parent.mkdir(parents=True, exist_ok=True)

with runtime_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(
        cfg,
        f,
        allow_unicode=True,
        sort_keys=False,
    )

print("Test runtime config generated:")
print(runtime_path)
print("Process       :", cfg["process"]["type"])
print("Representation:", cfg["representation"]["type"])
print("Sampler       :", cfg["sampler"]["name"])
print("Num steps     :", cfg["sampler"]["num_steps"])

PY


# ============================================================
# 读取最终 process / sampler 信息，仅用于打印
# ============================================================

readarray -t INFO < <(
python - "${TEST_RUNTIME_CONFIG}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print(cfg["process"]["type"])
print(cfg["representation"]["type"])
print(cfg["sampler"]["name"])
print(cfg["sampler"]["num_steps"])
PY
)

PROCESS_TYPE="${INFO[0]}"
REPRESENTATION_TYPE="${INFO[1]}"
SAMPLER_NAME="${INFO[2]}"
NUM_STEPS="${INFO[3]}"


# ============================================================
# 打印配置
# ============================================================

echo
echo "============================================================"
echo "Conditional DiT Evaluation"
echo "============================================================"
echo "Project dir      : ${PROJECT_DIR}"
echo "Physical GPU     : ${GPU_ID}"
echo "Device           : ${DEVICE}"
echo
echo "Experiment       : ${EXP_NAME}"
echo "Training config  : ${CONFIG_FILE}"
echo "Checkpoint       : ${CHECKPOINT_FILE}"
echo
echo "Process          : ${PROCESS_TYPE}"
echo "Representation   : ${REPRESENTATION_TYPE}"
echo "Sampler          : ${SAMPLER_NAME}"
echo "Num steps        : ${NUM_STEPS}"
echo
echo "Test manifest    : ${TEST_PAIR_LIST}"
echo "Test name        : ${TEST_NAME}"
echo "Batch size       : ${TEST_BATCH_SIZE}"
echo "Seed             : ${SAMPLER_SEED}"
echo
echo "Output dir       : ${OUTPUT_DIR}"
echo "Test config      : ${TEST_RUNTIME_CONFIG}"
echo "============================================================"
echo


# ============================================================
# 开始测试
# ============================================================

python evaluate.py \
    --config "${TEST_RUNTIME_CONFIG}" \
    --checkpoint "${CHECKPOINT_FILE}" \
    --split test \
    --output-dir "${OUTPUT_DIR}" \
    --seed "${SAMPLER_SEED}" \
    --device "${DEVICE}"


# ============================================================
# 输出结果
# ============================================================

echo
echo "============================================================"
echo "Evaluation finished"
echo "============================================================"
echo "Per-file metrics : ${OUTPUT_DIR}/per_file_metrics.csv"
echo "Summary metrics  : ${OUTPUT_DIR}/summary_metrics.json"
echo "Enhanced WAVs    : ${OUTPUT_DIR}/enhanced_wavs"
echo "Evaluation log   : ${OUTPUT_DIR}/evaluation.log"
echo "Runtime config   : ${TEST_RUNTIME_CONFIG}"
echo "============================================================"