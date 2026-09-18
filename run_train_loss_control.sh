#!/usr/bin/env bash

set -euo pipefail

GPU_ID=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ==================================================
# 基础配置
# ==================================================

BASE_CONFIG="configs/dccrn_base.yaml"

# 实验名称及输出目录
EXP_NAME="dccrn_snr_m15_0_si_snr=1.0"
OUTPUT_DIR="logs/${EXP_NAME}"

# 数据列表
TRAIN_CLEAN_LIST="data/ShipsEar-12class_W5H1/train_16k_signal.list"
TRAIN_NOISE_LIST="data/ShipsEar-12class_W5H1/train_16k_noise.list"
VALID_PAIR_LIST="data/ShipsEar-12class_W5H1/valid/list/valid_snr_-5dB.list"
TEST_PAIR_LIST="data/ShipsEar-12class_W5H1/test/list/valid_snr_-5dB.list"

# 数据参数
SAMPLE_RATE=16000
SEGMENT_SECONDS=5.0
SNR_MIN=-15
SNR_MAX=0
NUM_WORKERS=2

# 训练参数
SEED=42
BATCH_SIZE=32
EPOCHS=50
LEARNING_RATE=1e-4

# ==================================================
# 损失函数设置
# ==================================================
#
# 仅允许设置BASE_CONFIG中loss.components下已经存在的损失。
# 格式：损失名=权重
# 多个损失使用英文逗号分隔。
#
# 示例：
# LOSS_SPEC="si_snr=1.0"
# LOSS_SPEC="smse=1.0"
# LOSS_SPEC="si_snr=1.0,smse=0.1"
# LOSS_SPEC="waveform_mse=1.0"

LOSS_SPEC="si_snr=1.0"

# 最优模型监控指标
MONITOR_NAME="si_snr"
MONITOR_MODE="max"

# 留空表示从头训练
RESUME=""
# RESUME="logs/某个实验/checkpoints/last.pt"


# ==================================================
# 命令行参数
# ==================================================

show_help() {
    cat <<'EOF'
用法：
  bash run_train_loss_control.sh [选项]

选项：
  --loss SPEC
      设置训练损失。格式：损失名=权重
      多个损失使用英文逗号分隔。

  --monitor-name NAME
      设置最佳模型监控指标，例如si_snr或total_loss。

  --monitor-mode MODE
      设置监控方向，只能是max或min。

  -h, --help
      显示帮助。

示例：
  bash run_train_loss_control.sh --loss "si_snr=1.0"

  bash run_train_loss_control.sh \
      --loss "si_snr=1.0,smse=0.1"

  bash run_train_loss_control.sh \
      --loss "smse=1.0" \
      --monitor-name total_loss \
      --monitor-mode min
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --loss)
            [[ $# -ge 2 ]] || {
                echo "错误：--loss后必须提供参数" >&2
                exit 2
            }
            LOSS_SPEC="$2"
            shift 2
            ;;
        --monitor-name)
            [[ $# -ge 2 ]] || {
                echo "错误：--monitor-name后必须提供参数" >&2
                exit 2
            }
            MONITOR_NAME="$2"
            shift 2
            ;;
        --monitor-mode)
            [[ $# -ge 2 ]] || {
                echo "错误：--monitor-mode后必须提供参数" >&2
                exit 2
            }
            MONITOR_MODE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "错误：未知参数：$1" >&2
            show_help
            exit 2
            ;;
    esac
done

if [[ -z "${LOSS_SPEC//[[:space:]]/}" ]]; then
    echo "错误：LOSS_SPEC不能为空" >&2
    exit 2
fi

if [[ "${MONITOR_MODE}" != "max" && "${MONITOR_MODE}" != "min" ]]; then
    echo "错误：MONITOR_MODE只能是max或min" >&2
    exit 2
fi


# ==================================================
# 一般不需要修改下面内容
# ==================================================

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJECT_DIR}"

GENERATED_CONFIG="configs/generated/${EXP_NAME}.yaml"

mkdir -p "configs/generated"
mkdir -p "${OUTPUT_DIR}"

export BASE_CONFIG
export GENERATED_CONFIG
export EXP_NAME
export OUTPUT_DIR
export TRAIN_CLEAN_LIST
export TRAIN_NOISE_LIST
export VALID_PAIR_LIST
export TEST_PAIR_LIST
export SAMPLE_RATE
export SEGMENT_SECONDS
export SNR_MIN
export SNR_MAX
export NUM_WORKERS
export SEED
export BATCH_SIZE
export EPOCHS
export LEARNING_RATE
export LOSS_SPEC
export MONITOR_NAME
export MONITOR_MODE

python - <<'PY'
import math
import os
from pathlib import Path

import yaml


def get_required_mapping(parent, key, path):
    value = parent.get(key)

    if not isinstance(value, dict):
        raise ValueError(
            "{}必须存在，并且必须是YAML映射结构".format(path)
        )

    return value


base_config = Path(os.environ["BASE_CONFIG"])
output_config = Path(os.environ["GENERATED_CONFIG"])

with base_config.open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)

if not isinstance(config, dict):
    raise ValueError("基础YAML的顶层必须是映射结构")

# --------------------------------------------------
# 实验设置
# --------------------------------------------------

experiment = get_required_mapping(
    config,
    "experiment",
    "experiment",
)
experiment["name"] = os.environ["EXP_NAME"]
experiment["output_dir"] = os.environ["OUTPUT_DIR"]
experiment["seed"] = int(os.environ["SEED"])

# --------------------------------------------------
# 数据设置
# --------------------------------------------------

data = get_required_mapping(
    config,
    "data",
    "data",
)

data["train_clean_list"] = os.environ["TRAIN_CLEAN_LIST"]
data["train_noise_list"] = os.environ["TRAIN_NOISE_LIST"]
data["valid_pair_list"] = os.environ["VALID_PAIR_LIST"]
data["test_pair_list"] = os.environ["TEST_PAIR_LIST"]
data["sample_rate"] = int(os.environ["SAMPLE_RATE"])
data["segment_seconds"] = float(os.environ["SEGMENT_SECONDS"])
data["train_snr_min"] = float(os.environ["SNR_MIN"])
data["train_snr_max"] = float(os.environ["SNR_MAX"])
data["num_workers"] = int(os.environ["NUM_WORKERS"])

# --------------------------------------------------
# 训练设置
# --------------------------------------------------

training = get_required_mapping(
    config,
    "training",
    "training",
)

training["batch_size"] = int(os.environ["BATCH_SIZE"])
training["epochs"] = int(os.environ["EPOCHS"])
training["learning_rate"] = float(os.environ["LEARNING_RATE"])

# --------------------------------------------------
# 损失函数设置
# --------------------------------------------------

loss = get_required_mapping(
    config,
    "loss",
    "loss",
)
components = get_required_mapping(
    loss,
    "components",
    "loss.components",
)

if not components:
    raise ValueError("loss.components不能为空")

# 无论基础YAML中原来是否启用，先全部关闭。
for name, settings in components.items():
    if not isinstance(settings, dict):
        raise ValueError(
            "loss.components.{}必须是映射结构".format(name)
        )

    settings["enabled"] = False

requested = {}
loss_spec = os.environ["LOSS_SPEC"]

for raw_item in loss_spec.split(","):
    item = raw_item.strip()

    if not item:
        continue

    if "=" not in item:
        raise ValueError(
            "损失配置格式错误：{}；正确格式如si_snr=1.0".format(
                item
            )
        )

    name, weight_text = item.split("=", 1)
    name = name.strip()
    weight_text = weight_text.strip()

    if not name:
        raise ValueError("损失名称不能为空")

    if name in requested:
        raise ValueError(
            "损失名称重复：{}".format(name)
        )

    if name not in components:
        available = ", ".join(sorted(components))
        raise ValueError(
            "基础YAML中不存在损失项：{}；可用项：{}".format(
                name,
                available,
            )
        )

    try:
        weight = float(weight_text)
    except ValueError as error:
        raise ValueError(
            "损失{}的weight不是有效数字：{}".format(
                name,
                weight_text,
            )
        ) from error

    if not math.isfinite(weight):
        raise ValueError(
            "损失{}的weight必须是有限数字".format(name)
        )

    if weight <= 0:
        raise ValueError(
            "损失{}的weight必须大于0，当前为{}".format(
                name,
                weight,
            )
        )

    requested[name] = weight

if not requested:
    raise ValueError("至少需要指定一个有效损失")

# 只启用明确传入且权重合法的损失。
for name, weight in requested.items():
    components[name]["enabled"] = True
    components[name]["weight"] = weight

# --------------------------------------------------
# 监控指标
# --------------------------------------------------

monitor = loss.get("monitor")

if monitor is None:
    monitor = {}
    loss["monitor"] = monitor
elif not isinstance(monitor, dict):
    raise ValueError("loss.monitor必须是映射结构")

monitor_name = os.environ["MONITOR_NAME"].strip()
monitor_mode = os.environ["MONITOR_MODE"].strip()

if not monitor_name:
    raise ValueError("MONITOR_NAME不能为空")

if monitor_mode not in {"max", "min"}:
    raise ValueError("MONITOR_MODE只能是max或min")

monitor["name"] = monitor_name
monitor["mode"] = monitor_mode

# --------------------------------------------------
# 写入临时配置
# --------------------------------------------------

output_config.parent.mkdir(parents=True, exist_ok=True)

with output_config.open("w", encoding="utf-8") as stream:
    yaml.safe_dump(
        config,
        stream,
        allow_unicode=True,
        sort_keys=False,
    )

enabled_text = ", ".join(
    "{}={}".format(name, weight)
    for name, weight in requested.items()
)

print("生成配置文件：{}".format(output_config))
print("启用损失：{}".format(enabled_text))
print(
    "监控指标：{} ({})".format(
        monitor_name,
        monitor_mode,
    )
)
PY


echo "=========================================="
echo "实验名称：${EXP_NAME}"
echo "GPU：${GPU_ID}"
echo "训练SNR：${SNR_MIN}～${SNR_MAX} dB"
echo "Batch size：${BATCH_SIZE}"
echo "Epochs：${EPOCHS}"
echo "学习率：${LEARNING_RATE}"
echo "训练损失：${LOSS_SPEC}"
echo "监控指标：${MONITOR_NAME} (${MONITOR_MODE})"
echo "配置文件：${GENERATED_CONFIG}"
echo "输出目录：${OUTPUT_DIR}"
echo "=========================================="


if [[ -n "${RESUME}" ]]; then
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
