#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# 固定配置
# ============================================================

# SDE工作区，要求evaluate.py位于该目录下。
PROJECT_DIR="/data/huayifan/water_denoise/sgmse_workspace"

# 测试阶段当前只使用单张GPU。evaluate.py仅由主进程执行完整推理。
GPU_ID="1"

CONFIG_FILE="${PROJECT_DIR}/configs/sgmse_water_full_test.yaml"

# 修改为实际训练实验名称。
# EXP_NAME="sgmse_mixed_wp0.2_wr0.3_snr-15_0_seed42"
EXP_NAME="sgmse_mixed_wp0.2_wr0.3_snr-15_0_seed42"

# 可改为last.pt或某个epoch checkpoint。
CHECKPOINT_FILE="${PROJECT_DIR}/runs/${EXP_NAME}/checkpoints/best.pt"

# 测试数据：每行格式必须为 noisy<TAB>clean。
TEST_PAIR_LIST="/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/test/list/snr_-5dB.list"
# TEST_PAIR_LIST="/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/test/list/snr_0dB.list"
``
# 当前测试结果保存位置。
OUTPUT_DIR="${PROJECT_DIR}/runs/${EXP_NAME}/evaluation_best/snr_-5dB_steps=100_wo_corrector"

RUNTIME_CONFIG="${OUTPUT_DIR}/test_runtime.yaml"

# 增强音频保存数量：-1=保存全部，0=不保存，N=只保存前N个。
SAVE_AUDIO_LIMIT=100
# ============================================================
# SDE采样参数
# ============================================================

# Predictor-Corrector反向扩散步数。
NUM_STEPS=30  

# 每个预测步执行的corrector次数。
CORRECTOR_STEPS=1

# true：优先加载checkpoint中的EMA参数；通常测试时建议开启。
USE_EMA=true

# 固定推理随机种子，保证重复测试时采样过程可复现。
TEST_SEED=1234

# 生成式扩散推理显存和耗时较高，建议保持1。
TEST_BATCH_SIZE=1

# 测试时先设0最稳妥，避免DataLoader多进程带来额外问题。
NUM_WORKERS=0

# ============================================================
# 环境设置
# ============================================================

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED="${TEST_SEED}"

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "开始测试SGMSE/SDE模型"
echo "============================================================"
echo "项目目录：${PROJECT_DIR}"
echo "物理GPU：${GPU_ID}"
echo "配置文件：${CONFIG_FILE}"
echo "模型文件：${CHECKPOINT_FILE}"
echo "测试清单：${TEST_PAIR_LIST}"
echo "保存测试音频数：${SAVE_AUDIO_LIMIT}"
echo "输出目录：${OUTPUT_DIR}"
echo "采样步数：${NUM_STEPS}"
echo "Corrector次数：${CORRECTOR_STEPS}"
echo "使用EMA：${USE_EMA}"
echo "测试随机种子：${TEST_SEED}"
echo "测试Batch size：${TEST_BATCH_SIZE}"
echo "============================================================"

# ============================================================
# 文件与运行环境检查
# ============================================================

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "错误：配置文件不存在：${CONFIG_FILE}" >&2
    exit 1
fi

if [[ ! -f "${CHECKPOINT_FILE}" ]]; then
    echo "错误：模型文件不存在：${CHECKPOINT_FILE}" >&2
    exit 1
fi

if [[ ! -f "${TEST_PAIR_LIST}" ]]; then
    echo "错误：测试清单不存在：${TEST_PAIR_LIST}" >&2
    exit 1
fi

if [[ ! -f "${PROJECT_DIR}/evaluate.py" ]]; then
    echo "错误：找不到SDE测试入口：${PROJECT_DIR}/evaluate.py" >&2
    exit 1
fi
if ! [[ "${SAVE_AUDIO_LIMIT}" =~ ^-?[0-9]+$ ]]; then
    echo "错误：SAVE_AUDIO_LIMIT必须是整数" >&2
    exit 1
fi

if (( SAVE_AUDIO_LIMIT < -1 )); then
    echo "错误：SAVE_AUDIO_LIMIT只能是-1、0或正整数" >&2
    exit 1
fi

# 明确禁止CUDA不可用时静默回退到CPU。
python - <<PY
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "PyTorch当前无法使用CUDA，请检查Docker GPU映射、驱动和PyTorch版本"
    )

if torch.cuda.device_count() != 1:
    raise RuntimeError(
        "设置CUDA_VISIBLE_DEVICES后应只看到1张GPU，当前看到{}张".format(
            torch.cuda.device_count()
        )
    )

print("CUDA检查通过")
print("逻辑设备：cuda:0")
print("GPU名称：{}".format(torch.cuda.get_device_name(0)))
print("PyTorch：{}".format(torch.__version__))
print("PyTorch CUDA：{}".format(torch.version.cuda))
PY

# ============================================================
# 检查测试清单
# ============================================================

TEST_COUNT=$(awk '
    NF > 0 && $0 !~ /^[[:space:]]*#/ {
        count++
    }
    END {
        print count + 0
    }
' "${TEST_PAIR_LIST}")

if (( TEST_COUNT == 0 )); then
    echo "错误：测试清单为空" >&2
    exit 1
fi

echo "测试样本数：${TEST_COUNT}"

FORMAT_ERROR=$(awk -F '\t' '
    /^[[:space:]]*$/ {
        next
    }

    /^[[:space:]]*#/ {
        next
    }

    NF != 2 {
        print NR
        exit
    }
' "${TEST_PAIR_LIST}")

if [[ -n "${FORMAT_ERROR}" ]]; then
    echo "错误：测试清单第${FORMAT_ERROR}行格式不正确" >&2
    echo "正确格式：含噪音频路径<TAB>干净音频路径" >&2
    exit 1
fi

PROJECT_DIR="${PROJECT_DIR}" \
TEST_PAIR_LIST="${TEST_PAIR_LIST}" \
python - <<'PY'
import os
from collections import Counter
from pathlib import Path

list_path = Path(os.environ["TEST_PAIR_LIST"])
project_dir = Path(os.environ["PROJECT_DIR"])

missing = []
file_ids = []

with list_path.open("r", encoding="utf-8-sig") as stream:
    for line_number, raw_line in enumerate(stream, start=1):
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        fields = line.split("\t")
        if len(fields) != 2:
            continue

        noisy_text, clean_text = fields

        for path_text in (noisy_text, clean_text):
            path = Path(path_text)

            if not path.is_absolute():
                path = project_dir / path

            if not path.is_file():
                missing.append((line_number, str(path)))

        # evaluate.py使用含噪音频文件名stem作为增强音频名称。
        file_ids.append(Path(noisy_text).stem)

if missing:
    print("错误：测试清单中存在无效音频路径：")

    for line_number, path in missing[:20]:
        print("  第{}行：{}".format(line_number, path))

    if len(missing) > 20:
        print("  其余{}个路径未显示".format(len(missing) - 20))

    raise SystemExit(1)

duplicates = sorted(
    name
    for name, count in Counter(file_ids).items()
    if count > 1
)

if duplicates:
    print("错误：测试清单中存在重复的含噪文件名stem。")
    print("evaluate.py会用file_id.wav保存，重复名称将覆盖结果：")

    for name in duplicates[:20]:
        print("  {}".format(name))

    if len(duplicates) > 20:
        print("  其余{}个重复名称未显示".format(len(duplicates) - 20))

    raise SystemExit(1)

print("测试音频路径和输出文件名检查通过")
PY

# ============================================================
# 生成SDE测试专用配置
# ============================================================

export CONFIG_FILE
export RUNTIME_CONFIG
export TEST_PAIR_LIST
export TEST_BATCH_SIZE
export SAVE_AUDIO_LIMIT
export NUM_WORKERS
export NUM_STEPS
export CORRECTOR_STEPS
export USE_EMA
export TEST_SEED

python - <<'PY'
import os
from pathlib import Path

import yaml


config_path = Path(os.environ["CONFIG_FILE"])
runtime_path = Path(os.environ["RUNTIME_CONFIG"])

with config_path.open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)

if not isinstance(config, dict):
    raise ValueError("基础YAML顶层必须是映射结构")

for key in ("data", "training", "sampler", "validation"):
    if not isinstance(config.get(key), dict):
        raise ValueError("{}必须存在且为YAML映射结构".format(key))

# 当前测试清单。
config["data"]["test_manifest"] = os.environ["TEST_PAIR_LIST"]

# 测试完整音频，不按照训练窗长截断。
config["data"]["segment_validation"] = False
config["data"]["num_workers"] = int(os.environ["NUM_WORKERS"])

# 测试输出设置：-1保存全部，0不保存，N仅保存前N个。
save_audio_limit = int(os.environ["SAVE_AUDIO_LIMIT"])
if save_audio_limit < -1:
    raise ValueError("SAVE_AUDIO_LIMIT只能是-1、0或正整数")

config.setdefault("evaluation", {})
config["evaluation"]["save_audio_limit"] = save_audio_limit

# SDE生成式推理通常使用batch size 1。
config["training"]["batch_size"] = int(os.environ["TEST_BATCH_SIZE"])

# 固定反向扩散参数。
config["sampler"]["num_steps"] = int(os.environ["NUM_STEPS"])
config["sampler"]["corrector_steps"] = int(
    os.environ["CORRECTOR_STEPS"]
)
config["sampler"]["use_ema"] = (
    os.environ["USE_EMA"].strip().lower() == "true"
)

# evaluate.py从validation.seed读取测试随机种子。
config["validation"]["seed"] = int(os.environ["TEST_SEED"])

runtime_path.parent.mkdir(parents=True, exist_ok=True)

with runtime_path.open("w", encoding="utf-8") as stream:
    yaml.safe_dump(
        config,
        stream,
        allow_unicode=True,
        sort_keys=False,
    )

print("测试配置已生成：{}".format(runtime_path))
print("test_manifest：{}".format(config["data"]["test_manifest"]))
print("batch_size：{}".format(config["training"]["batch_size"]))
print("num_workers：{}".format(config["data"]["num_workers"]))
print("num_steps：{}".format(config["sampler"]["num_steps"]))
print(
    "corrector_steps：{}".format(
        config["sampler"]["corrector_steps"]
    )
)
print("use_ema：{}".format(config["sampler"]["use_ema"]))
print("seed：{}".format(config["validation"]["seed"]))
print("save_audio_limit：{}".format(config["evaluation"]["save_audio_limit"]))
PY

# ============================================================
# 开始测试
# ============================================================

# 清理上一次运行遗留的增强音频，避免旧WAV影响保存数量统计。
if [[ -d "${OUTPUT_DIR}/enhanced_wavs" ]]; then
    echo "清理旧增强音频目录：${OUTPUT_DIR}/enhanced_wavs"
    rm -rf "${OUTPUT_DIR}/enhanced_wavs"
fi

START_TIME=$(date +%s)

python -u evaluate.py \
    --config "${RUNTIME_CONFIG}" \
    --checkpoint "${CHECKPOINT_FILE}" \
    --split test \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
    2>&1 | tee "${OUTPUT_DIR}/console.log"

END_TIME=$(date +%s)
ELAPSED_SECONDS=$((END_TIME - START_TIME))

# ============================================================
# 检查并输出结果
# ============================================================

SUMMARY_FILE="${OUTPUT_DIR}/summary_metrics.json"
METRICS_FILE="${OUTPUT_DIR}/per_file_metrics.csv"
ENHANCED_DIR="${OUTPUT_DIR}/enhanced_wavs"
EVALUATION_LOG="${OUTPUT_DIR}/evaluation.log"

echo
echo "============================================================"
echo "SDE测试完成"
echo "============================================================"
echo "测试耗时：${ELAPSED_SECONDS}秒"
echo "汇总结果：${SUMMARY_FILE}"
echo "逐样本结果：${METRICS_FILE}"
echo "增强音频：${ENHANCED_DIR}"
echo "测试日志：${EVALUATION_LOG}"
echo "控制台日志：${OUTPUT_DIR}/console.log"
echo "运行配置：${RUNTIME_CONFIG}"
echo "============================================================"

for required_file in \
    "${SUMMARY_FILE}" \
    "${METRICS_FILE}" \
    "${EVALUATION_LOG}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "错误：测试结束后未生成：${required_file}" >&2
        exit 1
    fi
done

if [[ -d "${ENHANCED_DIR}" ]]; then
    ENHANCED_COUNT=$(
        find "${ENHANCED_DIR}" \
            -maxdepth 1 \
            -type f \
            -name '*.wav' \
            | wc -l
    )
else
    ENHANCED_COUNT=0
fi

if (( SAVE_AUDIO_LIMIT < 0 )); then
    EXPECTED_ENHANCED_COUNT=${TEST_COUNT}
elif (( SAVE_AUDIO_LIMIT > TEST_COUNT )); then
    EXPECTED_ENHANCED_COUNT=${TEST_COUNT}
else
    EXPECTED_ENHANCED_COUNT=${SAVE_AUDIO_LIMIT}
fi

echo "测试样本总数：${TEST_COUNT}"
echo "增强音频保存上限：${SAVE_AUDIO_LIMIT}"
echo "预期生成增强音频数：${EXPECTED_ENHANCED_COUNT}"
echo "实际生成增强音频数：${ENHANCED_COUNT}"

if (( ENHANCED_COUNT != EXPECTED_ENHANCED_COUNT )); then
    echo "警告：实际增强音频数量${ENHANCED_COUNT}与预期数量${EXPECTED_ENHANCED_COUNT}不一致" >&2
    echo "请确认sgmse/engine/evaluator.py已读取evaluation.save_audio_limit。" >&2
fi

echo
echo "前10个增强音频："
if [[ -d "${ENHANCED_DIR}" ]]; then
    find "${ENHANCED_DIR}" \
        -maxdepth 1 \
        -type f \
        -name '*.wav' \
        | sort \
        | head -10
else
    echo "（未保存增强音频）"
fi

echo
echo "测试指标汇总："
python -m json.tool "${SUMMARY_FILE}"