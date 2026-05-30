#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build-h20-nccl}"
MODEL_DIR="${1:-../models/Qwen3.5-0.8B}"
PP_SIZE="${PP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-2}"
WORLD_SIZE=$((PP_SIZE * TP_SIZE))
LAYERS="${LAYERS:-2}"
SEQ_LEN="${SEQ_LEN:-4}"
DTYPE="${DTYPE:-bfloat16}"
ID_PREFIX="${ID_PREFIX:-/tmp/cverl_qwen_pp_tp_train}"
NCCL_LIB_DIR="${NCCL_LIB_DIR:-}"

if [[ -n "${NCCL_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${NCCL_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

rm -f "${ID_PREFIX}"_*.bin /tmp/cverl_qwen_pp_tp_train_rank_*.log

for ((rank = 0; rank < WORLD_SIZE; ++rank)); do
  RANK="${rank}" \
  WORLD_SIZE="${WORLD_SIZE}" \
  LOCAL_RANK="${rank}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    "./${BUILD_DIR}/qwen3_5_pp_tp_train_smoke" "${MODEL_DIR}" \
      --global-rank "${rank}" \
      --pp-size "${PP_SIZE}" \
      --tp-size "${TP_SIZE}" \
      --device "${rank}" \
      --layers "${LAYERS}" \
      --seq-len "${SEQ_LEN}" \
      --dtype "${DTYPE}" \
      --id-prefix "${ID_PREFIX}" \
      >"/tmp/cverl_qwen_pp_tp_train_rank_${rank}.log" 2>&1 &
done

status=0
for job in $(jobs -p); do
  if ! wait "${job}"; then
    status=1
  fi
done

for ((rank = 0; rank < WORLD_SIZE; ++rank)); do
  echo "==== rank ${rank} ===="
  cat "/tmp/cverl_qwen_pp_tp_train_rank_${rank}.log"
done

exit "${status}"
