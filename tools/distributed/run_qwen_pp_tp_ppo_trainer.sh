#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-h20-nccl}"
MODEL_DIR="${1:-../models/Qwen3.5-0.8B}"
PP_SIZE="${PP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-2}"
WORLD_SIZE=$((PP_SIZE * TP_SIZE))
LAYERS="${LAYERS:-2}"
PROMPT_LEN="${PROMPT_LEN:-2}"
RESPONSE_LEN="${RESPONSE_LEN:-2}"
MICRO_BATCHES="${MICRO_BATCHES:-${PP_SIZE}}"
STEPS="${STEPS:-1}"
LR="${LR:-1e-8}"
CLIP_RATIO="${CLIP_RATIO:-0.2}"
ADVANTAGE_SCALE="${ADVANTAGE_SCALE:-1.0}"
MASTER_WEIGHTS="${MASTER_WEIGHTS:-true}"
SKIP_OPTIMIZER_STEP="${SKIP_OPTIMIZER_STEP:-false}"
VARY_TOKENS_BY_STEP="${VARY_TOKENS_BY_STEP:-false}"
JSONL_INPUT="${JSONL_INPUT:-}"
JSONL_MAX_EXAMPLES="${JSONL_MAX_EXAMPLES:-16}"
TOKENIZER_JSON="${TOKENIZER_JSON:-${MODEL_DIR}/tokenizer.json}"
DTYPE="${DTYPE:-bfloat16}"
ID_PREFIX="${ID_PREFIX:-/tmp/cverl_qwen_pp_tp_ppo}"
NCCL_LIB_DIR="${NCCL_LIB_DIR:-}"

if [[ -n "${NCCL_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${NCCL_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

if [[ "${BUILD_DIR}" = /* ]]; then
  TRAINER="${BUILD_DIR}/qwen3_5_pp_tp_ppo_trainer"
else
  TRAINER="./${BUILD_DIR}/qwen3_5_pp_tp_ppo_trainer"
fi

rm -f "${ID_PREFIX}"_*.bin /tmp/cverl_qwen_pp_tp_ppo_rank_*.log

for ((rank = 0; rank < WORLD_SIZE; ++rank)); do
  RANK="${rank}" \
  WORLD_SIZE="${WORLD_SIZE}" \
  LOCAL_RANK="${rank}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    "${TRAINER}" "${MODEL_DIR}" \
      --global-rank "${rank}" \
      --pp-size "${PP_SIZE}" \
      --tp-size "${TP_SIZE}" \
      --device "${rank}" \
      --layers "${LAYERS}" \
      --prompt-len "${PROMPT_LEN}" \
      --response-len "${RESPONSE_LEN}" \
      --micro-batches "${MICRO_BATCHES}" \
      --steps "${STEPS}" \
      --lr "${LR}" \
      --clip-ratio "${CLIP_RATIO}" \
      --advantage-scale "${ADVANTAGE_SCALE}" \
      --master-weights "${MASTER_WEIGHTS}" \
      --skip-optimizer-step "${SKIP_OPTIMIZER_STEP}" \
      --vary-tokens-by-step "${VARY_TOKENS_BY_STEP}" \
      --jsonl-input "${JSONL_INPUT}" \
      --jsonl-max-examples "${JSONL_MAX_EXAMPLES}" \
      --tokenizer-json "${TOKENIZER_JSON}" \
      --dtype "${DTYPE}" \
      --id-prefix "${ID_PREFIX}" \
      >"/tmp/cverl_qwen_pp_tp_ppo_rank_${rank}.log" 2>&1 &
done

status=0
for job in $(jobs -p); do
  if ! wait "${job}"; then
    status=1
  fi
done

for ((rank = 0; rank < WORLD_SIZE; ++rank)); do
  echo "==== rank ${rank} ===="
  cat "/tmp/cverl_qwen_pp_tp_ppo_rank_${rank}.log"
done

exit "${status}"
