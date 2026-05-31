#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-h20-nccl}"
MODEL_DIR="${1:-../models/Qwen3.5-0.8B}"
DP_SIZE="${DP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-2}"
CP_SIZE="${CP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-2}"
WORLD_SIZE=$((DP_SIZE * PP_SIZE * CP_SIZE * TP_SIZE))
LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE:-${WORLD_SIZE}}"
LAYERS="${LAYERS:-2}"
PROMPT_LEN="${PROMPT_LEN:-2}"
RESPONSE_LEN="${RESPONSE_LEN:-2}"
MICRO_BATCHES="${MICRO_BATCHES:-${PP_SIZE}}"
STEPS="${STEPS:-1}"
LR="${LR:-1e-8}"
CLIP_RATIO="${CLIP_RATIO:-0.2}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
DP_GRAD_BUCKET_MB="${DP_GRAD_BUCKET_MB:-25}"
TP_GRAD_BUCKET_MB="${TP_GRAD_BUCKET_MB:-${DP_GRAD_BUCKET_MB}}"
DP_GRAD_COMM_DTYPE="${DP_GRAD_COMM_DTYPE:-model}"
TP_GRAD_COMM_DTYPE="${TP_GRAD_COMM_DTYPE:-${DP_GRAD_COMM_DTYPE}}"
ADVANTAGE_SCALE="${ADVANTAGE_SCALE:-1.0}"
MASTER_WEIGHTS="${MASTER_WEIGHTS:-false}"
DP_SHARD_OPTIMIZER="${DP_SHARD_OPTIMIZER:-false}"
SKIP_OPTIMIZER_STEP="${SKIP_OPTIMIZER_STEP:-false}"
MEASURE_PARAM_DELTA="${MEASURE_PARAM_DELTA:-false}"
VARY_TOKENS_BY_STEP="${VARY_TOKENS_BY_STEP:-false}"
JSONL_INPUT="${JSONL_INPUT:-}"
ROLLOUT_JSON="${ROLLOUT_JSON:-}"
ROLLOUT_DIR="${ROLLOUT_DIR:-}"
ROLLOUT_IPC_JSON="${ROLLOUT_IPC_JSON:-}"
ROLLOUT_IPC_DIR="${ROLLOUT_IPC_DIR:-}"
ROLLOUT_NCCL_ID_FILE="${ROLLOUT_NCCL_ID_FILE:-}"
ROLLOUT_NCCL_WORLD_SIZE="${ROLLOUT_NCCL_WORLD_SIZE:-0}"
ROLLOUT_NCCL_RANK_OFFSET="${ROLLOUT_NCCL_RANK_OFFSET:-0}"
ROLLOUT_NCCL_SOURCE_RANK="${ROLLOUT_NCCL_SOURCE_RANK:-0}"
JSONL_MAX_EXAMPLES="${JSONL_MAX_EXAMPLES:-16}"
TOKENIZER_JSON="${TOKENIZER_JSON:-${MODEL_DIR}/tokenizer.json}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
METRICS_CSV="${METRICS_CSV:-}"
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
  local_rank=$((rank % LOCAL_WORLD_SIZE))
  RANK="${rank}" \
  WORLD_SIZE="${WORLD_SIZE}" \
  LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE}" \
  LOCAL_RANK="${local_rank}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    "${TRAINER}" "${MODEL_DIR}" \
      --global-rank "${rank}" \
      --dp-size "${DP_SIZE}" \
      --pp-size "${PP_SIZE}" \
      --cp-size "${CP_SIZE}" \
      --tp-size "${TP_SIZE}" \
      --device "${local_rank}" \
      --layers "${LAYERS}" \
      --prompt-len "${PROMPT_LEN}" \
      --response-len "${RESPONSE_LEN}" \
      --micro-batches "${MICRO_BATCHES}" \
      --steps "${STEPS}" \
      --lr "${LR}" \
      --clip-ratio "${CLIP_RATIO}" \
      --max-grad-norm "${MAX_GRAD_NORM}" \
      --dp-grad-bucket-mb "${DP_GRAD_BUCKET_MB}" \
      --tp-grad-bucket-mb "${TP_GRAD_BUCKET_MB}" \
      --dp-grad-comm-dtype "${DP_GRAD_COMM_DTYPE}" \
      --tp-grad-comm-dtype "${TP_GRAD_COMM_DTYPE}" \
      --advantage-scale "${ADVANTAGE_SCALE}" \
      --master-weights "${MASTER_WEIGHTS}" \
      --dp-shard-optimizer "${DP_SHARD_OPTIMIZER}" \
      --skip-optimizer-step "${SKIP_OPTIMIZER_STEP}" \
      --measure-param-delta "${MEASURE_PARAM_DELTA}" \
      --vary-tokens-by-step "${VARY_TOKENS_BY_STEP}" \
      --jsonl-input "${JSONL_INPUT}" \
      --rollout-json "${ROLLOUT_JSON}" \
      --rollout-dir "${ROLLOUT_DIR}" \
      --rollout-ipc-json "${ROLLOUT_IPC_JSON}" \
      --rollout-ipc-dir "${ROLLOUT_IPC_DIR}" \
      --rollout-nccl-id-file "${ROLLOUT_NCCL_ID_FILE}" \
      --rollout-nccl-world-size "${ROLLOUT_NCCL_WORLD_SIZE}" \
      --rollout-nccl-rank-offset "${ROLLOUT_NCCL_RANK_OFFSET}" \
      --rollout-nccl-source-rank "${ROLLOUT_NCCL_SOURCE_RANK}" \
      --jsonl-max-examples "${JSONL_MAX_EXAMPLES}" \
      --tokenizer-json "${TOKENIZER_JSON}" \
      --checkpoint-dir "${CHECKPOINT_DIR}" \
      --checkpoint-every "${CHECKPOINT_EVERY}" \
      --resume-checkpoint "${RESUME_CHECKPOINT}" \
      --metrics-csv "${METRICS_CSV}" \
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
