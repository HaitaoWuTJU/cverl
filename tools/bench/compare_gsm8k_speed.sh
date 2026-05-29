#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
VERL_DIR="${VERL_DIR:-${ROOT_DIR}/../verl}"
PYTHON="${PYTHON:-python3}"
DATASET_JSONL="${DATASET_JSONL:-${ROOT_DIR}/data/gsm8k-train-bench.jsonl}"
DATASET_ROWS="${DATASET_ROWS:-512}"
MODEL_DATASET_ID="${MODEL_DATASET_ID:-gsm8k}"
DEVICE="${DEVICE:-cpu}"
STEPS="${STEPS:-32}"
PROMPTS="${PROMPTS:-8}"
RESPONSES="${RESPONSES:-4}"
SEQ_LEN="${SEQ_LEN:-16}"
ACTION_DIM="${ACTION_DIM:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
PPO_EPOCHS="${PPO_EPOCHS:-2}"
LR="${LR:-0.003}"
SEED="${SEED:-17}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/build/bench}"
FORCE_DATASET="${FORCE_DATASET:-0}"

mkdir -p "${OUT_DIR}" "$(dirname "${DATASET_JSONL}")"

if [[ ! -x "${BUILD_DIR}/cverl_gsm8k_grpo_bench" ]]; then
  cmake --build "${BUILD_DIR}" --target cverl_gsm8k_grpo_bench
fi
if [[ ! -x "${BUILD_DIR}/hf_dataset_download" ]]; then
  cmake --build "${BUILD_DIR}" --target hf_dataset_download
fi

current_rows=0
if [[ -s "${DATASET_JSONL}" ]]; then
  current_rows="$(wc -l < "${DATASET_JSONL}" | tr -d ' ')"
fi
if [[ "${FORCE_DATASET}" = "1" || ! -s "${DATASET_JSONL}" || "${current_rows}" -lt "${DATASET_ROWS}" ]]; then
  "${BUILD_DIR}/hf_dataset_download" "${MODEL_DATASET_ID}" \
    --name main \
    --split train \
    --output-file "${DATASET_JSONL}" \
    --max-examples "${DATASET_ROWS}" \
    --python "${PYTHON}"
fi

COMMON_ARGS=(
  --dataset "${DATASET_JSONL}"
  --steps "${STEPS}"
  --prompts "${PROMPTS}"
  --responses "${RESPONSES}"
  --seq-len "${SEQ_LEN}"
  --action-dim "${ACTION_DIM}"
  --hidden-dim "${HIDDEN_DIM}"
  --ppo-epochs "${PPO_EPOCHS}"
  --lr "${LR}"
  --seed "${SEED}"
  --device "${DEVICE}"
)

echo "Running verl benchmark..."
PYTHONPATH="${VERL_DIR}:${PYTHONPATH:-}" "${PYTHON}" "${ROOT_DIR}/tools/bench/verl_gsm8k_grpo_bench.py" \
  "${COMMON_ARGS[@]}" | tee "${OUT_DIR}/verl_gsm8k_speed.csv"

echo "Running cverl benchmark..."
"${BUILD_DIR}/cverl_gsm8k_grpo_bench" "${COMMON_ARGS[@]}" | tee "${OUT_DIR}/cverl_gsm8k_speed.csv"

"${PYTHON}" - <<'PY' "${OUT_DIR}/verl_gsm8k_speed.csv" "${OUT_DIR}/cverl_gsm8k_speed.csv"
import csv
import sys

def read(path):
    with open(path, newline="") as f:
        return {row[0]: row[1] for row in csv.reader(f) if len(row) == 2}

verl = read(sys.argv[1])
cverl = read(sys.argv[2])
v = float(verl["train_tokens_per_second"])
c = float(cverl["train_tokens_per_second"])
print("summary_metric,value")
print(f"verl_train_tokens_per_second,{v:.6f}")
print(f"cverl_train_tokens_per_second,{c:.6f}")
print(f"cverl_vs_verl_speedup,{c / v:.6f}")
print(f"verl_seconds,{float(verl['seconds']):.6f}")
print(f"cverl_seconds,{float(cverl['seconds']):.6f}")
PY
