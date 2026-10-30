#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# EDIT THIS BLOCK ONLY
# ============================================================
PROJECT_DIR="/data/huayifan/water_denoise/dit_workspace"
BASE_CONFIG="configs/dit_score_complex_frame_token_1s.yaml"
GPU_IDS="0"
NUM_GPUS="1"                         # set 0 for CPU
MODE="log_mel"                       # log_mel | complex_plus_log_mel
PROCESS="flow"                       # flow | score
DATA_MODE="paired"                   # paired | on_the_fly

TRAIN_MANIFEST="/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/train/list/train_all_snr.list"
NOISE_MANIFEST=""                    # only on_the_fly: one noise path per line
VALID_MANIFEST="/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/test/list/test_all_snr.list"
TEST_MANIFEST="/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/test/list/test_all_snr.list"

HIDDEN="768"
DEPTH="10"
HEADS="12"
MLP_RATIO="2"
MEL_BANDS="128"
EPOCHS="100"
BATCH_SIZE="8"
LEARNING_RATE="0.0001"
AMP="true"
NUM_WORKERS="4"
SAMPLING_STEPS="10"                  # Flow: 10, Score: typically 30
STATS_MAX_ITEMS="0"                 # 0=all train samples; for a quick smoke use 50
SNR_MIN="-10"
SNR_MAX="-5"
EVAL_INTERVAL="5"
MAX_SAMPLING_BATCHES="4"
SEED="42"
EXP_NAME="${MODE}_${PROCESS}_h${HIDDEN}_d${DEPTH}_m${MEL_BANDS}_${DATA_MODE}"
RUN_ROOT="${PROJECT_DIR}/runs"
RESUME_CHECKPOINT=""
# ============================================================

cd "${PROJECT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
RUN_DIR="${RUN_ROOT}/${EXP_NAME}"
CONFIG_FILE="${RUN_DIR}/runtime_config.yaml"
mkdir -p "${RUN_DIR}"
for f in "${BASE_CONFIG}" "${TRAIN_MANIFEST}" "${VALID_MANIFEST}" "${TEST_MANIFEST}"; do
  [[ -f "${f}" ]] || { echo "Missing: ${f}" >&2; exit 1; }
done
if [[ "${DATA_MODE}" == "on_the_fly" ]]; then
  [[ -f "${NOISE_MANIFEST}" ]] || { echo "Missing noise manifest: ${NOISE_MANIFEST}" >&2; exit 1; }
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  [[ -f "${RESUME_CHECKPOINT}" ]] || { echo "Missing checkpoint: ${RESUME_CHECKPOINT}" >&2; exit 1; }
fi

python mel_scripts/make_mel_config.py \
  --base "${BASE_CONFIG}" --output "${CONFIG_FILE}" --name "${EXP_NAME}" \
  --mode "${MODE}" --process "${PROCESS}" --data-mode "${DATA_MODE}" \
  --train-manifest "${TRAIN_MANIFEST}" --valid-manifest "${VALID_MANIFEST}" \
  --test-manifest "${TEST_MANIFEST}" --noise-manifest "${NOISE_MANIFEST}" \
  --hidden "${HIDDEN}" --depth "${DEPTH}" --heads "${HEADS}" \
  --mlp-ratio "${MLP_RATIO}" --mel-bands "${MEL_BANDS}" \
  --batch "${BATCH_SIZE}" --epochs "${EPOCHS}" --lr "${LEARNING_RATE}" \
  --num-workers "${NUM_WORKERS}" --amp "${AMP}" --steps "${SAMPLING_STEPS}" \
  --sampling-valid-batches "${MAX_SAMPLING_BATCHES}" --eval-interval "${EVAL_INTERVAL}" \
  --snr-min "${SNR_MIN}" --snr-max "${SNR_MAX}" --seed "${SEED}"

# Mel normalization is computed exclusively on train waveforms and saved with SHA.
# When resuming, do not silently overwrite train stats or config metadata.
if [[ -z "${RESUME_CHECKPOINT}" ]]; then
  python mel_scripts/prepare_mel_stats.py --config "${CONFIG_FILE}" \
    --max-items "${STATS_MAX_ITEMS}" --seed "${SEED}"
else
  echo "Resume: reuse your previous runtime config and stats; this wrapper regenerates config!" >&2
  echo "Resume via: python train.py --config OLD_RUNTIME_CONFIG --resume CHECKPOINT" >&2
  exit 1
fi

python -m pytest -q tests/test_mel_extension.py
CMD=(python -u train.py --config "${CONFIG_FILE}" --num-gpus "${NUM_GPUS}")
"${CMD[@]}" 2>&1 | tee "${RUN_DIR}/console.log"
echo "Train finished: ${RUN_DIR}/checkpoints/best.pt"
