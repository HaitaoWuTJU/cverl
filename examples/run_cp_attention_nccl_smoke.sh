#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-h20-nccl}"
WORLD_SIZE="${WORLD_SIZE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
ID_FILE="${ID_FILE:-/tmp/cverl_cp_attention_nccl_id.bin}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCVERL_ENABLE_CUDA=ON \
  -DCVERL_ENABLE_NCCL=ON
cmake --build "${BUILD_DIR}" --target test_nccl_collectives -j "${BUILD_JOBS:-8}"

(
  cd "${ROOT_DIR}"
  NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
  NCCL_DEBUG="${NCCL_DEBUG}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  WORLD_SIZE="${WORLD_SIZE}" \
  ID_FILE="${ID_FILE}" \
    tools/distributed/run_nccl_smoke.sh "${BUILD_DIR}"
)
