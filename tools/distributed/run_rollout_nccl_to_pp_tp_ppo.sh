#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-h20-nccl}"
MODEL_DIR="${1:-../models/Qwen3.5-0.8B}"
PP_SIZE="${PP_SIZE:-3}"
TP_SIZE="${TP_SIZE:-1}"
TRAINER_WORLD_SIZE=$((PP_SIZE * TP_SIZE))
ROLLOUT_WORLD_SIZE="${ROLLOUT_WORLD_SIZE:-1}"
DATA_WORLD_SIZE=$((ROLLOUT_WORLD_SIZE + TRAINER_WORLD_SIZE))
ROLLOUT_SOURCE_RANK="${ROLLOUT_SOURCE_RANK:-0}"
ROLLOUT_DEVICE="${ROLLOUT_DEVICE:-${TRAINER_WORLD_SIZE}}"
ROLLOUT_NCCL_ID_FILE="${ROLLOUT_NCCL_ID_FILE:-/tmp/cverl_rollout_data_nccl.bin}"
PROMPT_LEN="${PROMPT_LEN:-32}"
RESPONSE_LEN="${RESPONSE_LEN:-8}"
ROLLOUT_BATCH="${ROLLOUT_BATCH:-4}"
STEPS="${STEPS:-1}"
NCCL_LIB_DIR="${NCCL_LIB_DIR:-}"

if [[ -n "${NCCL_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${NCCL_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

if [[ "${BUILD_DIR}" = /* ]]; then
  SOURCE_BIN="${BUILD_DIR}/nccl_rollout_batch_source"
  TRAINER_BIN="${BUILD_DIR}/qwen3_5_pp_tp_ppo_trainer"
else
  SOURCE_BIN="./${BUILD_DIR}/nccl_rollout_batch_source"
  TRAINER_BIN="./${BUILD_DIR}/qwen3_5_pp_tp_ppo_trainer"
fi

if [[ ! -x "${SOURCE_BIN}" || ! -x "${TRAINER_BIN}" ]]; then
  cmake --build "${BUILD_DIR}" --target nccl_rollout_batch_source qwen3_5_pp_tp_ppo_trainer -j "${BUILD_JOBS:-8}"
fi

targets=()
for ((pp = 0; pp < PP_SIZE; ++pp)); do
  if [[ "${pp}" -eq 0 || "${pp}" -eq $((PP_SIZE - 1)) ]]; then
    for ((tp = 0; tp < TP_SIZE; ++tp)); do
      trainer_rank=$((pp * TP_SIZE + tp))
      targets+=("$((ROLLOUT_WORLD_SIZE + trainer_rank))")
    done
  fi
done
TARGET_RANKS="$(IFS=,; echo "${targets[*]}")"

rm -f "${ROLLOUT_NCCL_ID_FILE}" /tmp/cverl_rollout_data_source.log

RANK="${ROLLOUT_SOURCE_RANK}" \
WORLD_SIZE="${DATA_WORLD_SIZE}" \
LOCAL_RANK="${ROLLOUT_DEVICE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  "${SOURCE_BIN}" \
    --rank "${ROLLOUT_SOURCE_RANK}" \
    --world-size "${DATA_WORLD_SIZE}" \
    --device "${ROLLOUT_DEVICE}" \
    --id-file "${ROLLOUT_NCCL_ID_FILE}" \
    --target-ranks "${TARGET_RANKS}" \
    --steps "${STEPS}" \
    --batch "${ROLLOUT_BATCH}" \
    --prompt-len "${PROMPT_LEN}" \
    --response-len "${RESPONSE_LEN}" \
    >"/tmp/cverl_rollout_data_source.log" 2>&1 &
source_pid=$!

status=0
ROLLOUT_NCCL_ID_FILE="${ROLLOUT_NCCL_ID_FILE}" \
ROLLOUT_NCCL_WORLD_SIZE="${DATA_WORLD_SIZE}" \
ROLLOUT_NCCL_RANK_OFFSET="${ROLLOUT_WORLD_SIZE}" \
ROLLOUT_NCCL_SOURCE_RANK="${ROLLOUT_SOURCE_RANK}" \
PP_SIZE="${PP_SIZE}" \
TP_SIZE="${TP_SIZE}" \
PROMPT_LEN="${PROMPT_LEN}" \
RESPONSE_LEN="${RESPONSE_LEN}" \
STEPS="${STEPS}" \
  tools/distributed/run_qwen_pp_tp_ppo_trainer.sh "${MODEL_DIR}" || status=$?

if ! wait "${source_pid}"; then
  status=1
fi

echo "==== rollout source ===="
cat /tmp/cverl_rollout_data_source.log

exit "${status}"
