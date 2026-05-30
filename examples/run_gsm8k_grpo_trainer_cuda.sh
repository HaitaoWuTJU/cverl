#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-h20-nccl}"
DATASET="${DATASET:-${ROOT_DIR}/data/gsm8k-train-smoke.jsonl}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/../models/Qwen3.5-0.8B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_DIR}/tokenizer.json}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCVERL_ENABLE_NCCL="${CVERL_ENABLE_NCCL:-ON}"
cmake --build "${BUILD_DIR}" -j "${JOBS:-2}" --target gsm8k_grpo_trainer

args=(
  --dataset "${DATASET}"
  --prompts "${PROMPTS:-2}"
  --n "${N:-2}"
  --steps "${STEPS:-1}"
  --ppo-epochs "${PPO_EPOCHS:-1}"
  --max-tokens "${MAX_TOKENS:-16}"
  --max-prompt-tokens "${MAX_PROMPT_TOKENS:-128}"
  --max-response-tokens "${MAX_RESPONSE_TOKENS:-32}"
  --policy qwen
  --model-dir "${MODEL_DIR}"
  --tokenizer hf
  --tokenizer-path "${TOKENIZER_PATH}"
  --device cuda
  --temperature "${TEMPERATURE:-1.0}"
)

if [[ -n "${QWEN_MAX_LAYERS:-}" ]]; then
  args+=(--qwen-max-layers "${QWEN_MAX_LAYERS}")
fi

"${BUILD_DIR}/gsm8k_grpo_trainer" "${args[@]}"
