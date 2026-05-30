#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
DATASET="${DATASET:-${ROOT_DIR}/data/gsm8k-train-smoke.jsonl}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" -j "${JOBS:-2}" --target gsm8k_grpo_trainer

"${BUILD_DIR}/gsm8k_grpo_trainer" \
  --dataset "${DATASET}" \
  --prompts "${PROMPTS:-2}" \
  --n "${N:-2}" \
  --steps "${STEPS:-2}" \
  --ppo-epochs "${PPO_EPOCHS:-1}" \
  --max-tokens "${MAX_TOKENS:-8}" \
  --max-prompt-tokens "${MAX_PROMPT_TOKENS:-64}" \
  --max-response-tokens "${MAX_RESPONSE_TOKENS:-16}" \
  --policy "${POLICY:-tiny}" \
  --tokenizer "${TOKENIZER:-byte}" \
  --device cpu \
  --temperature "${TEMPERATURE:-0}"
