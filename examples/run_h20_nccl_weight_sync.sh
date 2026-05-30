#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-h20-nccl}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCVERL_ENABLE_NCCL=ON
cmake --build "${BUILD_DIR}" -j "${JOBS:-2}" \
  --target test_nccl_collectives test_weight_sync test_rollout_worker

"${BUILD_DIR}/test_weight_sync"
"${BUILD_DIR}/test_rollout_worker"

NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth1}" \
NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
WORLD_SIZE="${WORLD_SIZE:-4}" \
"${ROOT_DIR}/tools/distributed/run_nccl_smoke.sh" "${BUILD_DIR}"
