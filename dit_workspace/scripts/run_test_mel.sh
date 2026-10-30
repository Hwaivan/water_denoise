#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="/data/huayifan/water_denoise/dit_workspace"
GPU_ID="0"
DEVICE="cuda"                       # cuda | cpu
EXP_NAME="log_mel_flow_h768_d10_m128_paired"  # match run_train_mel.sh
RUN_DIR="${PROJECT_DIR}/runs/${EXP_NAME}"
CHECKPOINT_FILE="${RUN_DIR}/checkpoints/best.pt"
TEST_MANIFEST="/data/huayifan/water_denoise/data/Zhaoyu/snr_-10_-5_class_all/test/list/test_all_snr.list"
SAMPLING_STEPS="10"                  # set 30/50/100 for sampler ablation
FLOW_SOLVER="euler"                  # euler | heun; ignored for score
USE_CORRECTOR="false"               # only score
TEST_TAG="all"

cd "${PROJECT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
CONFIG_FILE="${RUN_DIR}/runtime_config.yaml"
OUTPUT_DIR="${RUN_DIR}/evaluation/${TEST_TAG}_${SAMPLING_STEPS}steps"
TEST_CONFIG="${OUTPUT_DIR}/test_runtime.yaml"
for f in "${CONFIG_FILE}" "${CHECKPOINT_FILE}" "${TEST_MANIFEST}"; do
  [[ -f "${f}" ]] || { echo "Missing: ${f}" >&2; exit 1; }
done
mkdir -p "${OUTPUT_DIR}"
python - "${CONFIG_FILE}" "${TEST_CONFIG}" "${TEST_MANIFEST}" "${SAMPLING_STEPS}" "${FLOW_SOLVER}" "${USE_CORRECTOR}" <<'PY'
import sys, yaml
from pathlib import Path
src, dest, test_manifest, steps, solver, use_corrector = sys.argv[1:]
cfg = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
cfg["data"]["test_manifest"] = str(Path(test_manifest).resolve())
cfg["training"]["batch_size"] = 1
cfg["sampler"]["num_steps"] = int(steps)
if cfg["process"]["type"] == "flow":
    if solver not in ("euler", "heun"):
        raise ValueError("FLOW_SOLVER must be euler or heun")
    cfg["sampler"]["name"] = solver
else:
    cfg["sampler"]["corrector"] = "annealed_langevin" if use_corrector.lower() == "true" else "none"
    cfg["sampler"]["corrector_steps"] = 1 if use_corrector.lower() == "true" else 0
Path(dest).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print("Test:", cfg["representation"]["type"], cfg["process"]["type"], cfg["sampler"])
PY
python -u evaluate.py --config "${TEST_CONFIG}" \
  --checkpoint "${CHECKPOINT_FILE}" --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" 2>&1 | tee "${OUTPUT_DIR}/console.log"
echo "Summary: ${OUTPUT_DIR}/summary_metrics.json"
echo "WAVs:    ${OUTPUT_DIR}/enhanced_wavs/"
