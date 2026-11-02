#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Conditional DiT 串行多测试集评估
#
# 用法：
#   1. 修改“参数配置”区域；
#   2. 在 TEST_SETS 中添加任意数量测试集；
#   3. bash run_test_dit.sh
#
# 每个测试集会串行运行，并保存到独立目录。
# 任意一个测试失败时，默认停止后续测试。
# ============================================================


# ===================== 参数配置 =====================

PROJECT_DIR="/data/huayifan/water_denoise/dit_workspace"


# ------------------------------------------------------------
# GPU
# ------------------------------------------------------------

GPU_ID="1"
DEVICE="cuda"

# CPU:
# DEVICE="cpu"


# ------------------------------------------------------------
# 待测试的训练实验
# ------------------------------------------------------------

# EXP_NAME="dit_score_complex_frame_token_5s_onthefly"
# EXP_NAME="dit_score_complex_frame=256,seg=2s_onthefly"
# EXP_NAME="dit_score_complex_onthefly"
# EXP_NAME="dit_score_complex_frame_token_5s_hidSz=1024_real_onthefly"
# EXP_NAME="dit_score_complex_frame_token_5s_real_loss=sigma2_onthefly"
# EXP_NAME="dit_score_complex_frame_token_5s_hidSz=1024_real_loss=sigma2_onthefly"
# EXP_NAME="dit_score_complex_frame_token_5s_hidSz=1024_hop=128_real_loss=none_onthefly"
# EXP_NAME="dit_score_complex_frame_token_5s_hidSz=1024_hop=128_real_loss=sigma2_onthefly"
# EXP_NAME="dit_score_magnitude_frame_token_5s_hidSz=512_hop=100_real_loss=sigma2_onthefly"
# EXP_NAME="dit_score_complex_frame_token_5s_hidSz=1024_real_loss=sigma2_zys_paired"
# EXP_NAME="dit_score_log_magnitude_frame_token_5s_hidSz=768_hop=100_real_loss=sigma2_onthefly_withAMP"
# EXP_NAME="dit_score_complex_frame_token_5s_hidSz=1024_smallMLP_real_loss=sigma2_zys_paired"
# EXP_NAME="dit_score_complex_frame_token_5s_hidSz=1024_smallMLP_depth=8_real_loss=sigma2_zys_paired"

# EXP_NAME="dit_score_complex_frame_token_5s_hidSz=1024_smallMLP_real_LogSigma_loss=sigma2_zys_paired"

# EXP_NAME="dit_flow_complex_frame_token_5s_hidSz=1024_onthefly_-10~0dB"
# EXP_NAME="dit_flow_complex_frame_token_5s_hidSz=1024_numhead=8_smallMLP_loss=sigma2_zys_paired"
# EXP_NAME="dit_score_logmel_frame_token_5s_hidSz=768_numhead=12_smallMLP_loss=sigma2_zys_paired"
# EXP_NAME="dit_flow_logmel_frame_token_5s_hidSz=768_numhead=12_smallMLP_loss=sigma2_zys_paired"
# EXP_NAME="dit_flow_logmel_frame_token_5s_hidSz=512_numhead=16_smallMLP_loss=sigma2_zys_paired"
# EXP_NAME="dit_flow_logmel_frame_token_5s_hidSz=512_numhead=8_smallMLP_loss=sigma2_zys_paired"
# EXP_NAME="dit_flow_logmel_frame_token_5s_hidSz=512_numhead=8_smallMLP_loss=sigma2_zys_paired"
# EXP_NAME="dit_flow_logmel_frame_token_5s_hidSz=768_numhead=12_smallMLP_onethefly_-10~0dB"
EXP_NAME="dit_flow_logmel_wo_norm_frame_token_5s_hidSz=512_numhead=8_smallMLP_zys_paired"

# EXP_NAME="dit_score_complex_frame_token_5s_hidSz=1024_real_loss=sigma2_onthefly_-10~0dB"


RUN_ROOT="${PROJECT_DIR}/runs"
RUN_DIR="${RUN_ROOT}/${EXP_NAME}"

# 训练时保存的完整 runtime config
CONFIG_FILE="${RUN_DIR}/runtime_config.yaml"

# 默认测试 best checkpoint
CHECKPOINT_FILE="${RUN_DIR}/checkpoints/best.pt"

# 如需测试 last.pt：
# CHECKPOINT_FILE="${RUN_DIR}/checkpoints/last.pt"


# ------------------------------------------------------------
# 多测试集
#
# 格式：
#
#   "测试名称|pair list路径"
#
# 测试名称将直接作为保存目录名。
# 可以任意增加/删除。
# ------------------------------------------------------------

# TEST_SETS=(
#     "snr_-15dB_no_corrector|/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/test/list/snr_-15dB.list"
#     "snr_-10dB_no_corrector|/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/test/list/snr_-10dB.list"
#     "snr_-5dB_no_corrector|/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/test/list/snr_-5dB.list"
#     "snr_0dB_no_corrector|/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/test/list/snr_0dB.list"
# )

# TEST_SETS=(
#     "snr_-15dB_with_corrector|/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/test/list/snr_-15dB.list"
#     "snr_-10dB_with_corrector|/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/test/list/snr_-10dB.list"
#     "snr_-5dB_with_corrector|/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/test/list/snr_-5dB.list"
#     "snr_0dB_with_corrector|/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/test/list/snr_0dB.list"
# )

TEST_SETS=(
    "zys_test_all_no_corrector_30steps|/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/test/list/test_all_snr.list"
    # "zys_snr_-10dB_no_corrector|/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/test/list/snr_-10dB.list"
    # "zys_snr_-9dB_no_corrector|/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/test/list/snr_-9dB.list"
    # "zys_snr_-8dB_no_corrector|/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/test/list/snr_-8dB.list"
    # "zys_snr_-7dB_no_corrector|/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/test/list/snr_-7dB.list"
    # "zys_snr_-6dB_no_corrector|/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/test/list/snr_-6dB.list"
    # "zys_snr_-5dB_no_corrector|/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/test/list/snr_-5dB.list"
    "zys_train_snr_-5dB_no_corrector_30steps|/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/train/list/snr_-5dB.list"
)

# TEST_SETS=(
#     "zys_test_all_with_corrector_30steps|/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/test/list/test_all_snr.list"
#     # "zys_snr_-10dB_no_corrector|/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/test/list/snr_-10dB.list"
#     # "zys_snr_-9dB_no_corrector|/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/test/list/snr_-9dB.list"
#     # "zys_snr_-8dB_no_corrector|/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/test/list/snr_-8dB.list"
#     # "zys_snr_-7dB_no_corrector|/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/test/list/snr_-7dB.list"
#     # "zys_snr_-6dB_no_corrector|/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/test/list/snr_-6dB.list"
#     # "zys_snr_-5dB_no_corrector|/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/test/list/snr_-5dB.list"
#     "zys_train_snr_-5dB_with_corrector_30steps|/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/train/list/snr_-5dB.list"
# )


# TEST_SETS=(
#     "zys_test_all_with_corrector|/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/test/list/test_all_snr.list"
#     # "zys_snr_-10dB_with_corrector|/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/test/list/snr_-10dB.list"
#     # "zys_snr_-9dB_with_corrector|/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/test/list/snr_-9dB.list"
#     # "zys_snr_-8dB_with_corrector|/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/test/list/snr_-8dB.list"
#     # "zys_snr_-7dB_with_corrector|/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/test/list/snr_-7dB.list"
#     # "zys_snr_-6dB_with_corrector|/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/test/list/snr_-6dB.list"
#     # "zys_snr_-5dB_with_corrector|/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/test/list/snr_-5dB.list"
#     "zys_train_snr_-5dB_with_corrector|/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/train/list/snr_-5dB.list"
# )


# 例如还可以继续添加：
#
# TEST_SETS+=(
#     "snr_10dB|/data/.../snr_10dB.list"
# )


# ------------------------------------------------------------
# 测试输出根目录
# ------------------------------------------------------------

EVAL_ROOT="${RUN_DIR}/evaluation"


# ------------------------------------------------------------
# 测试参数
# ------------------------------------------------------------

TEST_BATCH_SIZE="1"
SAMPLER_SEED="1234"


# ------------------------------------------------------------
# Score corrector
# ------------------------------------------------------------

# true  : 使用 annealed Langevin corrector
# false : predictor-only
USE_CORRECTOR="false"

# 仅 USE_CORRECTOR=true 时生效
CORRECTOR_STEPS="1"
CORRECTOR_STEP_SIZE="0.5"

# ------------------------------------------------------------
# sampler override
#
# 留空：
#   完全使用训练 runtime_config.yaml
#
# Score:
#   pc
#
# DDPM:
#   ddpm
#
# Flow:
#   euler / heun
# ------------------------------------------------------------

# SAMPLER_OVERRIDE="pc"

SAMPLER_OVERRIDE="euler"
NUM_STEPS_OVERRIDE="30"

# Score 示例：
# SAMPLER_OVERRIDE="pc"
# NUM_STEPS_OVERRIDE="30"


# ===================== 参数配置结束 =====================



# ============================================================
# 初始化
# ============================================================

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"


# ============================================================
# 基础文件检查
# ============================================================

for f in \
    "${CONFIG_FILE}" \
    "${CHECKPOINT_FILE}"
do
    if [[ ! -f "${f}" ]]; then
        echo "[ERROR] file not found: ${f}" >&2
        exit 1
    fi
done


if [[ "${#TEST_SETS[@]}" -eq 0 ]]; then
    echo "[ERROR] TEST_SETS is empty." >&2
    exit 1
fi


# ============================================================
# 打印总体任务
# ============================================================

echo
echo "============================================================"
echo "Conditional DiT Multi-Test Evaluation"
echo "============================================================"
echo "Project dir      : ${PROJECT_DIR}"
echo "Experiment       : ${EXP_NAME}"
echo "Checkpoint       : ${CHECKPOINT_FILE}"
echo "Training config  : ${CONFIG_FILE}"
echo
echo "GPU              : ${GPU_ID}"
echo "Device           : ${DEVICE}"
echo "Batch size       : ${TEST_BATCH_SIZE}"
echo "Seed             : ${SAMPLER_SEED}"
echo
echo "Number of tests  : ${#TEST_SETS[@]}"
echo "Evaluation root  : ${EVAL_ROOT}"
echo "============================================================"
echo


# ============================================================
# 串行测试
# ============================================================

TOTAL_TESTS="${#TEST_SETS[@]}"
CURRENT_INDEX=0

for TEST_ITEM in "${TEST_SETS[@]}"; do

    CURRENT_INDEX=$((CURRENT_INDEX + 1))

    # --------------------------------------------------------
    # 解析：
    #   TEST_NAME|TEST_PAIR_LIST
    # --------------------------------------------------------

    IFS='|' read -r TEST_NAME TEST_PAIR_LIST <<< "${TEST_ITEM}"

    if [[ -z "${TEST_NAME}" || -z "${TEST_PAIR_LIST}" ]]; then
        echo "[ERROR] Invalid TEST_SETS entry:"
        echo "        ${TEST_ITEM}"
        echo "Expected:"
        echo '        "test_name|/path/to/test.list"'
        exit 1
    fi


    # --------------------------------------------------------
    # 检查测试 list
    # --------------------------------------------------------

    if [[ ! -f "${TEST_PAIR_LIST}" ]]; then
        echo "[ERROR] Test manifest not found:"
        echo "        ${TEST_PAIR_LIST}"
        exit 1
    fi


    # --------------------------------------------------------
    # 当前测试输出路径
    # --------------------------------------------------------

    OUTPUT_DIR="${EVAL_ROOT}/${TEST_NAME}"
    TEST_RUNTIME_CONFIG="${OUTPUT_DIR}/test_runtime.yaml"

    mkdir -p "${OUTPUT_DIR}"


    echo
    echo "============================================================"
    echo "[${CURRENT_INDEX}/${TOTAL_TESTS}] Testing: ${TEST_NAME}"
    echo "============================================================"
    echo "Test manifest    : ${TEST_PAIR_LIST}"
    echo "Output directory : ${OUTPUT_DIR}"
    echo "============================================================"
    echo


    # ========================================================
    # 生成该测试集专用 test_runtime.yaml
    # ========================================================

    # python - \
    #     "${CONFIG_FILE}" \
    #     "${TEST_RUNTIME_CONFIG}" \
    #     "${TEST_PAIR_LIST}" \
    #     "${TEST_BATCH_SIZE}" \
    #     "${SAMPLER_SEED}" \
    #     "${SAMPLER_OVERRIDE}" \
    #     "${NUM_STEPS_OVERRIDE}" <<'PY'

    python - \
        "${CONFIG_FILE}" \
        "${TEST_RUNTIME_CONFIG}" \
        "${TEST_PAIR_LIST}" \
        "${TEST_BATCH_SIZE}" \
        "${SAMPLER_SEED}" \
        "${SAMPLER_OVERRIDE}" \
        "${NUM_STEPS_OVERRIDE}" \
        "${USE_CORRECTOR}" \
        "${CORRECTOR_STEPS}" \
        "${CORRECTOR_STEP_SIZE}" <<'PY'

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
    use_corrector,
    corrector_steps,
    corrector_step_size,
) = sys.argv[1:]

# ------------------------------------------------------------
# Load training runtime config
# ------------------------------------------------------------

with open(config_file, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


# ------------------------------------------------------------
# Test dataset
# ------------------------------------------------------------

cfg["data"]["test_manifest"] = str(
    Path(test_manifest).resolve()
)

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

# ------------------------------------------------------------
# Score corrector override
# ------------------------------------------------------------

if process == "score":

    if use_corrector.lower() == "true":

        cfg["sampler"]["corrector"] = "annealed_langevin"
        cfg["sampler"]["corrector_steps"] = int(corrector_steps)
        cfg["sampler"]["corrector_step_size"] = float(corrector_step_size)

    elif use_corrector.lower() == "false":

        cfg["sampler"]["corrector"] = "none"
        cfg["sampler"]["corrector_steps"] = 0

    else:
        raise ValueError(
            f"USE_CORRECTOR must be true or false, got {use_corrector!r}"
        )

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

    # 当前 DDPM 需要完整 ancestral schedule
    if process == "ddpm":

        train_steps = int(
            cfg["process"]["ddpm"]["num_steps"]
        )

        if steps != train_steps:
            raise ValueError(
                f"Current DDPM sampler requires "
                f"num_steps={train_steps}, "
                f"but got {steps}. "
                "Reduced-step DDIM is not implemented."
            )

    cfg["sampler"]["num_steps"] = steps


# ------------------------------------------------------------
# Save test runtime config
# ------------------------------------------------------------

runtime_path = Path(runtime_file)
runtime_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with runtime_path.open(
    "w",
    encoding="utf-8",
) as f:

    yaml.safe_dump(
        cfg,
        f,
        allow_unicode=True,
        sort_keys=False,
    )


# ------------------------------------------------------------
# Print resolved test configuration
# ------------------------------------------------------------

print("Test runtime config generated:")
print(runtime_path)

print(
    "Tokenization  :",
    cfg["model"].get("tokenization", "patch2d")
)

print(
    "Process       :",
    cfg["process"]["type"]
)

print(
    "Representation:",
    cfg["representation"]["type"]
)

print(
    "Sampler       :",
    cfg["sampler"]["name"]
)

print(
    "Num steps     :",
    cfg["sampler"]["num_steps"]
)

print(
    "Corrector     :",
    cfg["sampler"].get("corrector", "none")
)

print(
    "Corrector step:",
    cfg["sampler"].get("corrector_steps", 0)
)

print(
    "Corrector size:",
    cfg["sampler"].get("corrector_step_size", 0.0)
)

print(
    "Chunk seconds :",
    cfg["inference"]["chunk_seconds"]
)

PY


    # ========================================================
    # 读取最终配置，仅用于显示
    # ========================================================

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
print(cfg["model"].get("tokenization", "patch2d"))
print(cfg["inference"]["chunk_seconds"])
PY
    )

    PROCESS_TYPE="${INFO[0]}"
    REPRESENTATION_TYPE="${INFO[1]}"
    SAMPLER_NAME="${INFO[2]}"
    NUM_STEPS="${INFO[3]}"
    TOKENIZATION="${INFO[4]}"
    CHUNK_SECONDS="${INFO[5]}"


    # ========================================================
    # 当前测试配置
    # ========================================================

    echo
    echo "------------------------------------------------------------"
    echo "Resolved test configuration"
    echo "------------------------------------------------------------"
    echo "Test             : ${TEST_NAME}"
    echo "Process          : ${PROCESS_TYPE}"
    echo "Representation   : ${REPRESENTATION_TYPE}"
    echo "Tokenization     : ${TOKENIZATION}"
    echo "Sampler          : ${SAMPLER_NAME}"
    echo "Num steps        : ${NUM_STEPS}"
    echo "Chunk seconds    : ${CHUNK_SECONDS}"
    echo "Batch size       : ${TEST_BATCH_SIZE}"
    echo "Seed             : ${SAMPLER_SEED}"
    echo "------------------------------------------------------------"
    echo


    # ========================================================
    # 开始当前测试
    # ========================================================

    python -u evaluate.py \
        --config "${TEST_RUNTIME_CONFIG}" \
        --checkpoint "${CHECKPOINT_FILE}" \
        --split test \
        --output-dir "${OUTPUT_DIR}" \
        --seed "${SAMPLER_SEED}" \
        --device "${DEVICE}"


    # ========================================================
    # 当前测试完成
    # ========================================================

    echo
    echo "------------------------------------------------------------"
    echo "[${CURRENT_INDEX}/${TOTAL_TESTS}] Finished: ${TEST_NAME}"
    echo "------------------------------------------------------------"
    echo "Per-file metrics : ${OUTPUT_DIR}/per_file_metrics.csv"
    echo "Summary metrics  : ${OUTPUT_DIR}/summary_metrics.json"
    echo "Enhanced WAVs    : ${OUTPUT_DIR}/enhanced_wavs"
    echo "Evaluation log   : ${OUTPUT_DIR}/evaluation.log"
    echo "Runtime config   : ${TEST_RUNTIME_CONFIG}"
    echo "------------------------------------------------------------"
    echo

done


# ============================================================
# 全部测试完成
# ============================================================

echo
echo "============================================================"
echo "All evaluation sets finished successfully."
echo "============================================================"

for TEST_ITEM in "${TEST_SETS[@]}"; do

    IFS='|' read -r TEST_NAME TEST_PAIR_LIST <<< "${TEST_ITEM}"

    echo "${TEST_NAME}:"
    echo "  ${EVAL_ROOT}/${TEST_NAME}/summary_metrics.json"

done

echo "============================================================"