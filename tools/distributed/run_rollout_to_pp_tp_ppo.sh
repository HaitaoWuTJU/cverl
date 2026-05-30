#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-h20-nccl}"
PYTHON="${PYTHON:-python}"
MODEL_DIR="${MODEL_DIR:-${1:-${ROOT_DIR}/../models/Qwen3.5-0.8B}}"
TOKENIZER_JSON="${TOKENIZER_JSON:-${MODEL_DIR}/tokenizer.json}"
DATASET="${DATASET:-${ROOT_DIR}/data/gsm8k-train-bench.jsonl}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${ROOT_DIR}/build/rollout-pp-tp-batches}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_DIR}}"

PP_SIZE="${PP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-2}"
MICRO_BATCHES="${MICRO_BATCHES:-4}"
LAYERS="${LAYERS:-24}"
PROMPT_LEN="${PROMPT_LEN:-128}"
RESPONSE_LEN="${RESPONSE_LEN:-64}"
STEPS="${STEPS:-5}"
PROMPTS="${PROMPTS:-2}"
N="${N:-2}"
MAX_TOKENS="${MAX_TOKENS:-64}"
LR="${LR:-1e-10}"
DTYPE="${DTYPE:-bfloat16}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth1}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/build/pp-tp-ppo-checkpoints}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-0}"
METRICS_CSV="${METRICS_CSV:-${ROOT_DIR}/build/metrics/pp_tp_ppo_metrics.csv}"

mkdir -p "${ROLLOUT_DIR}" "$(dirname "${METRICS_CSV}")"

if [[ ! -x "${BUILD_DIR}/qwen3_5_pp_tp_ppo_trainer" ]]; then
  cmake --build "${BUILD_DIR}" --target qwen3_5_pp_tp_ppo_trainer -j "${BUILD_JOBS:-8}"
fi

rm -f "${ROLLOUT_DIR}"/rollout_step_*.json

"${PYTHON}" "${ROOT_DIR}/tools/rollout/vllm_online_grpo.py" \
  --dataset "${DATASET}" \
  --model-dir "${MODEL_DIR}" \
  --tokenizer-path "${TOKENIZER_JSON}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --base-url "${VLLM_BASE_URL}" \
  --rollout-backend "${ROLLOUT_BACKEND}" \
  --skip-train \
  --dump-rollout-dir "${ROLLOUT_DIR}" \
  --steps "${STEPS}" \
  --prompts "${PROMPTS}" \
  --n "${N}" \
  --max-tokens "${MAX_TOKENS}" \
  --max-prompt-tokens "${PROMPT_LEN}" \
  --max-response-tokens "${RESPONSE_LEN}"

(
  cd "${ROOT_DIR}"
  NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
  NCCL_DEBUG="${NCCL_DEBUG}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  BUILD_DIR="${BUILD_DIR}" \
  PP_SIZE="${PP_SIZE}" \
  TP_SIZE="${TP_SIZE}" \
  MICRO_BATCHES="${MICRO_BATCHES}" \
  LAYERS="${LAYERS}" \
  PROMPT_LEN="${PROMPT_LEN}" \
  RESPONSE_LEN="${RESPONSE_LEN}" \
  STEPS="${STEPS}" \
  DTYPE="${DTYPE}" \
  LR="${LR}" \
  ROLLOUT_DIR="${ROLLOUT_DIR}" \
  TOKENIZER_JSON="${TOKENIZER_JSON}" \
  CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
  CHECKPOINT_EVERY="${CHECKPOINT_EVERY}" \
  METRICS_CSV="${METRICS_CSV}" \
    tools/distributed/run_qwen_pp_tp_ppo_trainer.sh "${MODEL_DIR}"
)

echo "metrics_csv=${METRICS_CSV}"
if [[ "${CHECKPOINT_EVERY}" != "0" ]]; then
  echo "checkpoint_dir=${CHECKPOINT_DIR}"
fi
