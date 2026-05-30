#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build-h20-nccl}"
WORLD_SIZE="${WORLD_SIZE:-2}"
ID_FILE="${ID_FILE:-/tmp/cverl_nccl_gpu_batch_id.bin}"
NCCL_LIB_DIR="${NCCL_LIB_DIR:-}"

if [[ -n "${NCCL_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${NCCL_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

if [[ "${BUILD_DIR}" = /* ]]; then
  TEST_BIN="${BUILD_DIR}/test_nccl_gpu_batch_transport"
else
  TEST_BIN="./${BUILD_DIR}/test_nccl_gpu_batch_transport"
fi

rm -f "${ID_FILE}" /tmp/cverl_nccl_gpu_batch_rank_*.log

for ((rank = 0; rank < WORLD_SIZE; ++rank)); do
  RANK="${rank}" \
  WORLD_SIZE="${WORLD_SIZE}" \
  LOCAL_RANK="${rank}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
    "${TEST_BIN}" \
      --rank "${rank}" \
      --world-size "${WORLD_SIZE}" \
      --device "${rank}" \
      --id-file "${ID_FILE}" \
      >"/tmp/cverl_nccl_gpu_batch_rank_${rank}.log" 2>&1 &
done

status=0
for job in $(jobs -p); do
  if ! wait "${job}"; then
    status=1
  fi
done

for ((rank = 0; rank < WORLD_SIZE; ++rank)); do
  echo "==== rank ${rank} ===="
  cat "/tmp/cverl_nccl_gpu_batch_rank_${rank}.log"
done

exit "${status}"
