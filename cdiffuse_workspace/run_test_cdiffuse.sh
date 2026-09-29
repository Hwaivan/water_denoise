#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# CDiffuSE 测试脚本
# 只需要修改“参数配置”区域
# ============================================================

# ===================== 参数配置 =====================

PROJECT_DIR="/data/huayifan/water_denoise/cdiffuse_workspace"
GPU_ID="1"

# 训练实验名称
EXP_NAME="cdiffuse_water_base_modified"
RUN_DIR="${PROJECT_DIR}/runs/${EXP_NAME}"

# 优先使用训练时保存的 runtime_config，保证测试与训练参数一致
CONFIG_FILE="${RUN_DIR}/runtime_config.yaml"
CHECKPOINT_FILE="${RUN_DIR}/checkpoints/best.pt"

# 测试集：每行格式 noisy<TAB>clean
TEST_PAIR_LIST="/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/test/list/snr_-5dB.list"
# TEST_PAIR_LIST="/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/test/list/snr_0dB.list"

# 本次测试名称
TEST_NAME="snr_-5dB_real"
# TEST_NAME="snr_0dB"
OUTPUT_DIR="${RUN_DIR}/evaluation/${TEST_NAME}"
RUNTIME_CONFIG="${OUTPUT_DIR}/test_runtime.yaml"

# 测试 batch size
TEST_BATCH_SIZE="1"
SAMPLER_SEED="1234"

# ---------------- evaluation 参数 ----------------

# final = (1-ratio) * raw_enhanced + ratio * noisy
# 公平比较模型原始输出时建议设为 0.0
NOISY_MIX_RATIO="0.0"

# 最终增强音频保存数量：
# -1 = 全部保存；0 = 不保存；N = 保存前 N 条
SAVE_AUDIO_LIMIT="-1"

# 是否保存反向扩散过程中的中间时间步音频
SAVE_DIFFUSION_STEPS="true"

# 只对前多少条测试样本保存中间状态：
# -1 = 全部；0 = 不保存；N = 前 N 条
DIFFUSION_EXAMPLE_LIMIT="100"

# 每隔多少个反向扩散时间步保存一次
# Base 模型 T=50，设为 5 时保存：50,45,...,5,1
DIFFUSION_STEP_INTERVAL="5"

# ============================================================

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# ============================================================
# 文件检查
# ============================================================

for f in "${CONFIG_FILE}" "${CHECKPOINT_FILE}" "${TEST_PAIR_LIST}"; do
    if [[ ! -f "${f}" ]]; then
        echo "错误：文件不存在：${f}" >&2
        exit 1
    fi
done

if (( DIFFUSION_STEP_INTERVAL < 1 )); then
    echo "错误：DIFFUSION_STEP_INTERVAL 必须 >= 1" >&2
    exit 1
fi

# ============================================================
# 生成测试专用 runtime YAML
# ============================================================

python - <<PY
from pathlib import Path
import yaml

config_path = Path(r"${CONFIG_FILE}")
runtime_path = Path(r"${RUNTIME_CONFIG}")

with config_path.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 测试数据
config["data"]["test_manifest"] = r"${TEST_PAIR_LIST}"

# 当前 CDiffuSE 的 build_dataloader 共用 training.batch_size
config["training"]["batch_size"] = int("${TEST_BATCH_SIZE}")

# sampler
config["sampler"]["seed"] = int("${SAMPLER_SEED}")

# evaluation
config.setdefault("evaluation", {})
config["evaluation"]["noisy_mix_ratio"] = float("${NOISY_MIX_RATIO}")
config["evaluation"]["save_audio_limit"] = int("${SAVE_AUDIO_LIMIT}")
config["evaluation"]["save_diffusion_steps"] = "${SAVE_DIFFUSION_STEPS}".lower() == "true"
config["evaluation"]["diffusion_example_limit"] = int("${DIFFUSION_EXAMPLE_LIMIT}")
config["evaluation"]["diffusion_step_interval"] = int("${DIFFUSION_STEP_INTERVAL}")

runtime_path.parent.mkdir(parents=True, exist_ok=True)
with runtime_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

print("测试配置已生成：", runtime_path)
print("总扩散时间步：", config["diffusion"]["num_steps"])
PY

# ============================================================
# 打印配置
# ============================================================

echo "============================================================"
echo "开始测试 CDiffuSE"
echo "============================================================"
echo "项目目录               ：${PROJECT_DIR}"
echo "物理 GPU                ：${GPU_ID}"
echo "配置文件                ：${CONFIG_FILE}"
echo "模型文件                ：${CHECKPOINT_FILE}"
echo "测试清单                ：${TEST_PAIR_LIST}"
echo "输出目录                ：${OUTPUT_DIR}"
echo "batch size              ：${TEST_BATCH_SIZE}"
echo "sampler seed            ：${SAMPLER_SEED}"
echo "noisy mix ratio         ：${NOISY_MIX_RATIO}"
echo "save audio limit        ：${SAVE_AUDIO_LIMIT}"
echo "save diffusion steps    ：${SAVE_DIFFUSION_STEPS}"
echo "diffusion example limit ：${DIFFUSION_EXAMPLE_LIMIT}"
echo "diffusion step interval ：${DIFFUSION_STEP_INTERVAL}"
echo "============================================================"

# ============================================================
# 开始测试
# ============================================================

python evaluate.py \
    --config "${RUNTIME_CONFIG}" \
    --checkpoint "${CHECKPOINT_FILE}" \
    --split test \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda

# ============================================================
# 输出结果
# ============================================================

echo
echo "============================================================"
echo "测试完成"
echo "============================================================"
echo "逐样本指标：${OUTPUT_DIR}/per_file_metrics.csv"
echo "汇总指标  ：${OUTPUT_DIR}/summary_metrics.json"
echo "最终音频  ：${OUTPUT_DIR}/enhanced_wavs"
echo "原始输出  ：${OUTPUT_DIR}/raw_enhanced_wavs"
echo "过程时间步：${OUTPUT_DIR}/diffusion_steps"
echo "运行配置  ：${RUNTIME_CONFIG}"
echo "============================================================"
