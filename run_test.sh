#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# 固定配置
# ============================================================

PROJECT_DIR="/data/huayifan/water_denoise"
GPU_ID="1"

CONFIG_FILE="${PROJECT_DIR}/configs/dccrn_base.yaml"

CHECKPOINT_FILE="${PROJECT_DIR}/logs/dccrn_snr_m10_10/checkpoints/best.pt"

OUTPUT_DIR="${PROJECT_DIR}/logs/dccrn_snr_m10_10/evaluation_best_true/snr_0dB"

# TEST_PAIR_LIST="${PROJECT_DIR}/data/ShipsEar-12class_W5H1/test/list/valid_snr_-5dB.list"
# TEST_PAIR_LIST="${PROJECT_DIR}/data/ShipsEar-12class_W5H1/test/list/valid_snr_-5dB.list"
TEST_PAIR_LIST="${PROJECT_DIR}/data/ShipsEar-12class_W5H1/test/list/snr_0dB.list"

RUNTIME_CONFIG="${OUTPUT_DIR}/test_runtime.yaml"

# ============================================================
# 环境设置
# ============================================================

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "开始测试DCCRN模型"
echo "============================================================"
echo "项目目录：${PROJECT_DIR}"
echo "物理GPU：${GPU_ID}"
echo "配置文件：${CONFIG_FILE}"
echo "模型文件：${CHECKPOINT_FILE}"
echo "测试清单：${TEST_PAIR_LIST}"
echo "输出目录：${OUTPUT_DIR}"
echo "============================================================"

# ============================================================
# 文件检查
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

# 统计非空、非注释样本数
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

# ============================================================
# 检查清单格式：noisy<TAB>clean
# ============================================================

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

# ============================================================
# 检查音频文件
# ============================================================

python - <<PY
from pathlib import Path

list_path = Path("${TEST_PAIR_LIST}")
project_dir = Path("${PROJECT_DIR}")

missing = []

with list_path.open("r", encoding="utf-8-sig") as f:
    for line_number, line in enumerate(f, start=1):
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        fields = line.split("\t")
        if len(fields) != 2:
            continue

        for path_text in fields:
            path = Path(path_text)

            # 相对路径默认相对于项目根目录
            if not path.is_absolute():
                path = project_dir / path

            if not path.is_file():
                missing.append((line_number, str(path)))

if missing:
    print("错误：测试清单中存在无效音频路径：")

    for line_number, path in missing[:20]:
        print(f"  第{line_number}行：{path}")

    if len(missing) > 20:
        print(f"  其余{len(missing) - 20}个路径未显示")

    raise SystemExit(1)

print("测试音频路径检查通过")
PY

# ============================================================
# 生成测试专用临时配置
# ============================================================

python - <<PY
from pathlib import Path
import yaml

config_path = Path("${CONFIG_FILE}")
runtime_path = Path("${RUNTIME_CONFIG}")

with config_path.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 指定当前测试清单
config["data"]["test_pair_list"] = "${TEST_PAIR_LIST}"

# 长音频测试建议batch size为1
config["training"]["batch_size"] = 1

# 测试不需要AMP配置，但保留原配置不会影响evaluate.py
with runtime_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(
        config,
        f,
        allow_unicode=True,
        sort_keys=False,
    )

print("测试配置已生成：${RUNTIME_CONFIG}")
print("测试batch size：1")
PY

# ============================================================
# 开始测试
# ============================================================

python evaluate.py \
    --config "${RUNTIME_CONFIG}" \
    --checkpoint "${CHECKPOINT_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda

# ============================================================
# 输出结果
# ============================================================

echo
echo "============================================================"
echo "测试完成"
echo "============================================================"
echo "汇总结果：${OUTPUT_DIR}/summary.json"
echo "逐样本结果：${OUTPUT_DIR}/per_sample.csv"
echo "音频示例：${OUTPUT_DIR}/audio_examples"
echo "运行配置：${RUNTIME_CONFIG}"
echo "============================================================"

if [[ -f "${OUTPUT_DIR}/summary.json" ]]; then
    echo
    echo "测试指标汇总："
    python -m json.tool "${OUTPUT_DIR}/summary.json"
fi