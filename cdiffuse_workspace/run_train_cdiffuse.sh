#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# ===================== 参数配置 =====================
GPU_ID="0"

BASE_CONFIG="configs/cdiffuse_water_base.yaml"
# BASE_CONFIG="configs/cdiffuse_water_large.yaml"

EXP_NAME="cdiffuse_water_base"
RUN_DIR="runs/${EXP_NAME}"

TRAIN_MANIFEST="/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/train_16k_signal.list"
VALID_MANIFEST="/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/valid/list/valid_snr_-5dB.list"
NOISE_MANIFEST="/data/huayifan/water_denoise/data/ShipsEar-12class_W5H1/train_16k_noise.list"

SNR_MIN="-15"
SNR_MAX="0"

SEED="42"
EPOCHS="160"
BATCH_SIZE="4"
LEARNING_RATE="0.0002"
NUM_WORKERS="4"
AMP="true"

# Full reverse-diffusion validation, aligned with sgmse_workspace.
EVAL_INTERVAL="5"
MAX_SAMPLING_BATCHES="2"
VALIDATION_SEED="1234"
SAVE_EVERY="10"

# 留空表示从头训练；断点续训时填写 last.pt
RESUME_CHECKPOINT=""
# ==================================================

RUNTIME_CONFIG="${RUN_DIR}/runtime_config.yaml"
CONSOLE_LOG="${RUN_DIR}/console.log"

for f in "${BASE_CONFIG}" "${TRAIN_MANIFEST}" "${VALID_MANIFEST}" "${NOISE_MANIFEST}"; do
    [[ -f "${f}" ]] || { echo "[ERROR] file not found: ${f}"; exit 1; }
done

if [[ -n "${RESUME_CHECKPOINT}" && ! -f "${RESUME_CHECKPOINT}" ]]; then
    echo "[ERROR] resume checkpoint not found: ${RESUME_CHECKPOINT}"
    exit 1
fi

mkdir -p "${RUN_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

python - \
    "${BASE_CONFIG}" "${RUNTIME_CONFIG}" "${EXP_NAME}" "${RUN_DIR}" \
    "${TRAIN_MANIFEST}" "${VALID_MANIFEST}" "${NOISE_MANIFEST}" \
    "${SNR_MIN}" "${SNR_MAX}" "${SEED}" "${EPOCHS}" "${BATCH_SIZE}" \
    "${LEARNING_RATE}" "${NUM_WORKERS}" "${AMP}" \
    "${EVAL_INTERVAL}" "${MAX_SAMPLING_BATCHES}" "${VALIDATION_SEED}" "${SAVE_EVERY}" <<'PY'
import sys
from pathlib import Path
import yaml

(
    base_config, runtime_config, exp_name, run_dir,
    train_manifest, valid_manifest, noise_manifest,
    snr_min, snr_max, seed, epochs, batch_size,
    learning_rate, num_workers, amp,
    eval_interval, max_sampling_batches, validation_seed, save_every,
) = sys.argv[1:]

with open(base_config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["project"]["name"] = exp_name
cfg["project"]["output_dir"] = str(Path(run_dir).resolve())

cfg["data"]["train_manifest"] = str(Path(train_manifest).resolve())
cfg["data"]["valid_manifest"] = str(Path(valid_manifest).resolve())
cfg["data"]["noise_manifest"] = str(Path(noise_manifest).resolve())
cfg["data"]["snr_min"] = float(snr_min)
cfg["data"]["snr_max"] = float(snr_max)
cfg["data"]["num_workers"] = int(num_workers)

cfg["training"]["seed"] = int(seed)
cfg["training"]["epochs"] = int(epochs)
cfg["training"]["batch_size"] = int(batch_size)
cfg["training"]["learning_rate"] = float(learning_rate)
cfg["training"]["amp"] = amp.lower() == "true"
cfg["training"]["save_every"] = int(save_every)

cfg.setdefault("validation", {})
cfg["validation"]["eval_interval"] = int(eval_interval)
cfg["validation"]["max_sampling_batches"] = int(max_sampling_batches)
cfg["validation"]["seed"] = int(validation_seed)
cfg["validation"]["best_metric"] = "si_snri"

cfg.setdefault("logging", {})
cfg["logging"].setdefault("tensorboard", True)
cfg["logging"].setdefault("jsonl", True)

Path(runtime_config).parent.mkdir(parents=True, exist_ok=True)
with open(runtime_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY

echo "============================================================"
echo "CDiffuSE Training"
echo "GPU                    : ${GPU_ID}"
echo "Base config            : ${BASE_CONFIG}"
echo "Experiment             : ${EXP_NAME}"
echo "Train manifest         : ${TRAIN_MANIFEST}"
echo "Valid manifest         : ${VALID_MANIFEST}"
echo "Noise manifest         : ${NOISE_MANIFEST}"
echo "SNR                    : ${SNR_MIN} ~ ${SNR_MAX} dB"
echo "Epochs                 : ${EPOCHS}"
echo "Batch size             : ${BATCH_SIZE}"
echo "Eval interval          : ${EVAL_INTERVAL}"
echo "Sampling valid batches : ${MAX_SAMPLING_BATCHES}"
echo "Runtime config         : ${RUNTIME_CONFIG}"
echo "Resume                 : ${RESUME_CHECKPOINT:-None}"
echo "Python training log    : ${RUN_DIR}/training.log"
echo "Structured metrics     : ${RUN_DIR}/metrics.jsonl"
echo "Console log            : ${CONSOLE_LOG}"
echo "============================================================"

CMD=(python -u train.py --config "${RUNTIME_CONFIG}" --device cuda)
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    CMD+=(--resume "${RESUME_CHECKPOINT}")
fi

"${CMD[@]}" 2>&1 | tee "${CONSOLE_LOG}"

echo "============================================================"
echo "Training finished."
echo "Best : ${RUN_DIR}/checkpoints/best.pt"
echo "Last : ${RUN_DIR}/checkpoints/last.pt"
echo "Log  : ${RUN_DIR}/training.log"
echo "JSONL: ${RUN_DIR}/metrics.jsonl"
echo "TB   : ${RUN_DIR}/tensorboard"
echo "============================================================"
