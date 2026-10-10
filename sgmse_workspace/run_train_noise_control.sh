#!/usr/bin/env bash

set -euo pipefail

# ==================================================
# 基础配置
# ==================================================

GPU_ID=0
DEVICE="cuda"

# 为确定性 CUDA 运算提供约束。
export CUBLAS_WORKSPACE_CONFIG=:4096:8

BASE_CONFIG="configs/sgmse_water_full.yaml"

# 留空时根据噪声参数、SNR和随机种子自动生成。
EXP_NAME=""
OUTPUT_DIR=""

# 数据列表
TRAIN_CLEAN_LIST="/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/train_16k_signal.list"
TRAIN_NOISE_LIST="/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/train_16k_noise.list"
VALID_PAIR_LIST="/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/valid/list/valid_snr_-5dB.list"
TEST_PAIR_LIST="/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/test/list/valid_snr_-5dB.list"


# 数据参数
SAMPLE_RATE=16000
# SEGMENT_SECONDS=2.048
SEGMENT_SECONDS=2.5
NUM_WORKERS=4

# ==================================================
# 环境加噪设置
# ==================================================
#
# noise_mode支持：
#   real  ：纯真实噪声
#   white ：纯白高斯噪声
#   mixed ：概率控制的混合噪声
#
# mixed模式：
# 1. 以WHITE_NOISE_PROBABILITY概率使用纯白噪声；
# 2. 否则使用“真实噪声+白噪声”；
# 3. WHITE_NOISE_RATIO控制第二个分支中白噪声的功率比例；
# 4. 最终总噪声统一按SNR_MIN～SNR_MAX缩放。
NOISE_MODE="mixed"
WHITE_NOISE_PROBABILITY=0.2
WHITE_NOISE_RATIO=0.3

SNR_MIN=-15
SNR_MAX=0

# ==================================================
# 训练参数
# ==================================================

SEED=42
DETERMINISTIC=true
BATCH_SIZE=6
EPOCHS=160
LEARNING_RATE=1e-4

# 留空表示从头训练
RESUME=""
# RESUME="runs/某个实验/checkpoints/last.pt"


# ==================================================
# 命令行参数
# ==================================================

show_help() {
    cat <<'EOF'
用法：
  bash run_train_noise_control.sh [选项]

选项：
  --noise-mode MODE
      加噪模式：real、white或mixed。

  --white-probability VALUE
      mixed模式下使用纯白噪声的概率，范围[0,1]。

  --white-ratio VALUE
      mixed模式进入真实噪声分支后，混入白噪声的功率比例，
      范围[0,1]。

  --snr-min VALUE
      训练最小SNR，单位dB。

  --snr-max VALUE
      训练最大SNR，单位dB。

  --seed VALUE
      设置随机种子。

  --gpu VALUE
      设置GPU编号，例如0。

  --device DEVICE
      设置运行设备：cuda或cpu。

  --batch-size VALUE
      设置batch size。

  --epochs VALUE
      设置训练轮数。

  --learning-rate VALUE
      设置学习率。

  --exp-name NAME
      手动指定实验名称。留空时自动生成。

  --output-dir PATH
      手动指定输出目录。留空时使用runs/实验名称。

  --resume PATH
      从指定checkpoint恢复训练。

  -h, --help
      显示帮助。

示例：
  # 纯真实噪声
  bash run_train_noise_control.sh \
      --noise-mode real \
      --snr-min -5 \
      --snr-max 20

  # 纯白噪声
  bash run_train_noise_control.sh \
      --noise-mode white \
      --snr-min -5 \
      --snr-max 20

  # 混合噪声
  bash run_train_noise_control.sh \
      --noise-mode mixed \
      --white-probability 0.2 \
      --white-ratio 0.3 \
      --snr-min -5 \
      --snr-max 20 \
      --seed 42

  # 恢复训练
  bash run_train_noise_control.sh \
      --exp-name sgmse_mixed_wp0.2_wr0.3_snr-5_20_seed42 \
      --resume runs/sgmse_mixed_wp0.2_wr0.3_snr-5_20_seed42/checkpoints/last.pt
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --noise-mode)
            [[ $# -ge 2 ]] || {
                echo "错误：--noise-mode后必须提供参数" >&2
                exit 2
            }
            NOISE_MODE="$2"
            shift 2
            ;;
        --white-probability)
            [[ $# -ge 2 ]] || {
                echo "错误：--white-probability后必须提供参数" >&2
                exit 2
            }
            WHITE_NOISE_PROBABILITY="$2"
            shift 2
            ;;
        --white-ratio)
            [[ $# -ge 2 ]] || {
                echo "错误：--white-ratio后必须提供参数" >&2
                exit 2
            }
            WHITE_NOISE_RATIO="$2"
            shift 2
            ;;
        --snr-min)
            [[ $# -ge 2 ]] || {
                echo "错误：--snr-min后必须提供参数" >&2
                exit 2
            }
            SNR_MIN="$2"
            shift 2
            ;;
        --snr-max)
            [[ $# -ge 2 ]] || {
                echo "错误：--snr-max后必须提供参数" >&2
                exit 2
            }
            SNR_MAX="$2"
            shift 2
            ;;
        --seed)
            [[ $# -ge 2 ]] || {
                echo "错误：--seed后必须提供参数" >&2
                exit 2
            }
            SEED="$2"
            shift 2
            ;;
        --gpu)
            [[ $# -ge 2 ]] || {
                echo "错误：--gpu后必须提供参数" >&2
                exit 2
            }
            GPU_ID="$2"
            shift 2
            ;;
        --device)
            [[ $# -ge 2 ]] || {
                echo "错误：--device后必须提供参数" >&2
                exit 2
            }
            DEVICE="$2"
            shift 2
            ;;
        --batch-size)
            [[ $# -ge 2 ]] || {
                echo "错误：--batch-size后必须提供参数" >&2
                exit 2
            }
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            [[ $# -ge 2 ]] || {
                echo "错误：--epochs后必须提供参数" >&2
                exit 2
            }
            EPOCHS="$2"
            shift 2
            ;;
        --learning-rate)
            [[ $# -ge 2 ]] || {
                echo "错误：--learning-rate后必须提供参数" >&2
                exit 2
            }
            LEARNING_RATE="$2"
            shift 2
            ;;
        --exp-name)
            [[ $# -ge 2 ]] || {
                echo "错误：--exp-name后必须提供参数" >&2
                exit 2
            }
            EXP_NAME="$2"
            shift 2
            ;;
        --output-dir)
            [[ $# -ge 2 ]] || {
                echo "错误：--output-dir后必须提供参数" >&2
                exit 2
            }
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --resume)
            [[ $# -ge 2 ]] || {
                echo "错误：--resume后必须提供参数" >&2
                exit 2
            }
            RESUME="$2"
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


# ==================================================
# 参数检查
# ==================================================

if [[ "${NOISE_MODE}" != "real" \
   && "${NOISE_MODE}" != "white" \
   && "${NOISE_MODE}" != "mixed" ]]; then
    echo "错误：NOISE_MODE只能是real、white或mixed" >&2
    exit 2
fi

if [[ "${DEVICE}" != "cuda" && "${DEVICE}" != "cpu" ]]; then
    echo "错误：DEVICE只能是cuda或cpu" >&2
    exit 2
fi

if [[ "${DEVICE}" == "cuda" ]]; then
    if [[ ! "${GPU_ID}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "错误：--gpu 应为 0 或 0,1,2,3 形式" >&2
        exit 2
    fi

    IFS=',' read -r -a GPU_ARRAY <<< "${GPU_ID}"
    GPU_COUNT="${#GPU_ARRAY[@]}"
else
    GPU_COUNT=0
fi

if [[ "${DETERMINISTIC}" != "true" \
   && "${DETERMINISTIC}" != "false" ]]; then
    echo "错误：DETERMINISTIC只能是true或false" >&2
    exit 2
fi

if [[ -z "${EXP_NAME}" ]]; then
    EXP_NAME="sgmse_${NOISE_MODE}_wp${WHITE_NOISE_PROBABILITY}_wr${WHITE_NOISE_RATIO}_snr${SNR_MIN}_${SNR_MAX}_seed${SEED}"
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="runs/${EXP_NAME}"
fi


# ==================================================
# 一般不需要修改下面内容
# ==================================================

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJECT_DIR}"

if [[ ! -f "${BASE_CONFIG}" ]]; then
    echo "错误：找不到基础配置文件：${BASE_CONFIG}" >&2
    exit 2
fi

if [[ ! -f "train.py" ]]; then
    echo "错误：当前目录下找不到train.py" >&2
    exit 2
fi

if [[ -n "${RESUME}" && ! -f "${RESUME}" ]]; then
    echo "错误：找不到恢复训练文件：${RESUME}" >&2
    exit 2
fi

GENERATED_CONFIG="configs/generated/${EXP_NAME}.yaml"

mkdir -p "configs/generated"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/reproducibility"

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
export NUM_WORKERS
export NOISE_MODE
export WHITE_NOISE_PROBABILITY
export WHITE_NOISE_RATIO
export SNR_MIN
export SNR_MAX
export SEED
export DETERMINISTIC
export BATCH_SIZE
export EPOCHS
export LEARNING_RATE

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


def get_finite_float(name):
    value = float(os.environ[name])

    if not math.isfinite(value):
        raise ValueError("{}必须是有限数字".format(name))

    return value


base_config = Path(os.environ["BASE_CONFIG"])
output_config = Path(os.environ["GENERATED_CONFIG"])

with base_config.open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)

if not isinstance(config, dict):
    raise ValueError("基础YAML的顶层必须是映射结构")

# --------------------------------------------------
# 参数检查
# --------------------------------------------------

noise_mode = os.environ["NOISE_MODE"].strip().lower()

if noise_mode not in {"real", "white", "mixed"}:
    raise ValueError(
        "NOISE_MODE只能是real、white或mixed"
    )

white_probability = get_finite_float(
    "WHITE_NOISE_PROBABILITY"
)
white_ratio = get_finite_float(
    "WHITE_NOISE_RATIO"
)
snr_min = get_finite_float("SNR_MIN")
snr_max = get_finite_float("SNR_MAX")

if not 0.0 <= white_probability <= 1.0:
    raise ValueError(
        "WHITE_NOISE_PROBABILITY必须位于[0,1]"
    )

if not 0.0 <= white_ratio <= 1.0:
    raise ValueError(
        "WHITE_NOISE_RATIO必须位于[0,1]"
    )

if snr_min > snr_max:
    raise ValueError(
        "SNR_MIN不能大于SNR_MAX"
    )

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
experiment["deterministic"] = (
    os.environ["DETERMINISTIC"].lower() == "true"
)

# --------------------------------------------------
# 数据设置
# --------------------------------------------------

data = get_required_mapping(
    config,
    "data",
    "data",
)

data["train_manifest"] = os.environ["TRAIN_CLEAN_LIST"]
data["noise_manifest"] = os.environ["TRAIN_NOISE_LIST"]
data["valid_manifest"] = os.environ["VALID_PAIR_LIST"]
data["test_manifest"] = os.environ["TEST_PAIR_LIST"]

data["sample_rate"] = int(os.environ["SAMPLE_RATE"])
data["segment_seconds"] = float(
    os.environ["SEGMENT_SECONDS"]
)
data["num_workers"] = int(os.environ["NUM_WORKERS"])

data["noise_mode"] = noise_mode
data["white_noise_probability"] = white_probability
data["white_noise_ratio"] = white_ratio
data["snr_min"] = snr_min
data["snr_max"] = snr_max

if noise_mode in {"real", "mixed"}:
    if not data["noise_manifest"].strip():
        raise ValueError(
            "real和mixed模式必须提供noise_manifest"
        )

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
training["learning_rate"] = float(
    os.environ["LEARNING_RATE"]
)

# resume由train.py的--resume参数传入，避免基础YAML污染。
training["resume"] = None

# --------------------------------------------------
# 写入生成配置
# --------------------------------------------------

output_config.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with output_config.open("w", encoding="utf-8") as stream:
    yaml.safe_dump(
        config,
        stream,
        allow_unicode=True,
        sort_keys=False,
    )

print("生成配置文件：{}".format(output_config))
print("加噪模式：{}".format(noise_mode))
print(
    "纯白噪声概率：{}".format(
        white_probability
    )
)
print(
    "混合分支白噪声比例：{}".format(
        white_ratio
    )
)
print(
    "训练SNR：{}～{} dB".format(
        snr_min,
        snr_max,
    )
)
PY


# ==================================================
# 保存复现信息
# ==================================================

cp "${GENERATED_CONFIG}" \
   "${OUTPUT_DIR}/reproducibility/config.resolved.yaml"
cp "$0" \
   "${OUTPUT_DIR}/reproducibility/launch_script.sh"

if git rev-parse --is-inside-work-tree \
    >/dev/null 2>&1; then
    git rev-parse HEAD \
        > "${OUTPUT_DIR}/reproducibility/git_commit.txt"
    git status --short \
        > "${OUTPUT_DIR}/reproducibility/git_status.txt"
    git diff \
        > "${OUTPUT_DIR}/reproducibility/git_diff.patch"
fi

python --version \
    > "${OUTPUT_DIR}/reproducibility/python_version.txt" \
    2>&1

python -m pip freeze \
    > "${OUTPUT_DIR}/reproducibility/pip_freeze.txt" \
    2>/dev/null || true

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi \
        > "${OUTPUT_DIR}/reproducibility/nvidia_smi.txt" \
        2>&1 || true
fi


echo "=========================================="
echo "实验名称：${EXP_NAME}"
echo "运行设备：${DEVICE}"
echo "GPU：${GPU_ID}"
echo "加噪模式：${NOISE_MODE}"
echo "纯白噪声概率：${WHITE_NOISE_PROBABILITY}"
echo "混合分支白噪声比例：${WHITE_NOISE_RATIO}"
echo "训练SNR：${SNR_MIN}～${SNR_MAX} dB"
echo "随机种子：${SEED}"
echo "确定性训练：${DETERMINISTIC}"
echo "Batch size：${BATCH_SIZE}"
echo "Epochs：${EPOCHS}"
echo "学习率：${LEARNING_RATE}"
echo "配置文件：${GENERATED_CONFIG}"
echo "输出目录：${OUTPUT_DIR}"
echo "=========================================="


# ==================================================
# 启动训练
# ==================================================

TRAIN_COMMAND=(
    python -u train.py
    --config "${GENERATED_CONFIG}"
    --device "${DEVICE}"
    --num-gpus "${GPU_COUNT}"
)

if [[ -n "${RESUME}" ]]; then
    TRAIN_COMMAND+=(--resume "${RESUME}")
fi

{
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU_ID}"
    printf 'PYTHONHASHSEED=%q ' "${SEED}"
    printf '%q ' "${TRAIN_COMMAND[@]}"
    printf '\n'
} > "${OUTPUT_DIR}/reproducibility/command.txt"

if [[ "${DEVICE}" == "cuda" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTHONHASHSEED="${SEED}" \
    "${TRAIN_COMMAND[@]}" \
        2>&1 | tee "${OUTPUT_DIR}/console.log"
else
    PYTHONHASHSEED="${SEED}" \
    "${TRAIN_COMMAND[@]}" \
        2>&1 | tee "${OUTPUT_DIR}/console.log"
fi
