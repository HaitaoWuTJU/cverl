#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-h20-nccl}"
DATASET="${DATASET:-${ROOT_DIR}/data/gsm8k-train-smoke.jsonl}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/../models/Qwen3.5-0.8B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_DIR}/tokenizer.json}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCVERL_ENABLE_NCCL="${CVERL_ENABLE_NCCL:-ON}" \
  -DCVERL_BUILD_PYTHON_BRIDGE=ON
cmake --build "${BUILD_DIR}" -j "${JOBS:-2}" --target _cverl_vllm_bridge

args=(
  --dataset "${DATASET}"
  --model-dir "${MODEL_DIR}"
  --tokenizer-path "${TOKENIZER_PATH}"
  --served-model-name "${SERVED_MODEL_NAME:-${MODEL_DIR}}"
  --base-url "${VLLM_BASE_URL:-http://127.0.0.1:8000}"
  --device "${DEVICE:-cuda:0}"
  --param-dtype "${PARAM_DTYPE:-bfloat16}"
  --qwen-max-layers "${QWEN_MAX_LAYERS:--1}"
  --rollout-backend "${ROLLOUT_BACKEND:-vllm}"
  --steps "${STEPS:-8}"
  --prompts "${PROMPTS:-4}"
  --n "${N:-4}"
  --max-tokens "${MAX_TOKENS:-256}"
  --max-prompt-tokens "${MAX_PROMPT_TOKENS:-256}"
  --max-response-tokens "${MAX_RESPONSE_TOKENS:-256}"
  --ppo-epochs "${PPO_EPOCHS:-1}"
  --lr "${LR:-3e-6}"
  --temperature "${TEMPERATURE:-1.0}"
  --top-p "${TOP_P:-1.0}"
  --master-address "${MASTER_ADDRESS:-127.0.0.1}"
  --master-port "${MASTER_PORT:-29577}"
  --world-size "${WORLD_SIZE:-2}"
)

if [[ "${NO_WEIGHT_SYNC:-0}" == "1" ]]; then
  args+=(--no-weight-sync)
fi
if [[ "${SYNC_INITIAL:-0}" == "1" ]]; then
  args+=(--sync-initial)
fi
if [[ "${MEASURE_PARAM_DELTA:-0}" == "1" ]]; then
  args+=(--measure-param-delta)
fi

PYTHONPATH="${BUILD_DIR}:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"${ROOT_DIR}/tools/rollout/vllm_online_grpo.py" "${args[@]}"
