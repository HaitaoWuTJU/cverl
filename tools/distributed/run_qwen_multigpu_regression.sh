#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-h20-nccl}"
MODEL_DIR="${1:-${MODEL_DIR:-${ROOT_DIR}/../models/Qwen3.5-0.8B}}"
JSONL_INPUT="${JSONL_INPUT:-${ROOT_DIR}/data/gsm8k-train-bench.jsonl}"
JSONL_MAX_EXAMPLES="${JSONL_MAX_EXAMPLES:-8}"
TOKENIZER_JSON="${TOKENIZER_JSON:-${MODEL_DIR}/tokenizer.json}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/build/gpu-smoke}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth1}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
DTYPE="${DTYPE:-bfloat16}"
LAYERS="${LAYERS:-4}"
PROMPT_LEN="${PROMPT_LEN:-2}"
RESPONSE_LEN="${RESPONSE_LEN:-2}"
STEPS="${STEPS:-5}"
ADVANTAGE_SCALE="${ADVANTAGE_SCALE:-0.05}"
LR="${LR:-1e-10}"
RUN_BASIC_TESTS="${RUN_BASIC_TESTS:-1}"

mkdir -p "${OUT_DIR}"

require_file() {
  local path="$1"
  local what="$2"
  if [[ ! -e "${path}" ]]; then
    echo "missing ${what}: ${path}" >&2
    exit 2
  fi
}

require_exec() {
  local path="$1"
  if [[ ! -x "${path}" ]]; then
    echo "building missing target for ${path}" >&2
    cmake --build "${BUILD_DIR}" --target "$(basename "${path}")" -j "${BUILD_JOBS:-8}"
  fi
}

run_case() {
  local name="$1"
  local pp="$2"
  local tp="$3"
  local micro_batches="$4"
  local log="${OUT_DIR}/${name}.log"

  echo "running ${name}: PP=${pp} TP=${tp} micro_batches=${micro_batches} steps=${STEPS}"
  (
    cd "${ROOT_DIR}"
    NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
    NCCL_DEBUG="${NCCL_DEBUG}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    BUILD_DIR="${BUILD_DIR}" \
    PP_SIZE="${pp}" \
    TP_SIZE="${tp}" \
    LAYERS="${LAYERS}" \
    PROMPT_LEN="${PROMPT_LEN}" \
    RESPONSE_LEN="${RESPONSE_LEN}" \
    MICRO_BATCHES="${micro_batches}" \
    STEPS="${STEPS}" \
    DTYPE="${DTYPE}" \
    ADVANTAGE_SCALE="${ADVANTAGE_SCALE}" \
    LR="${LR}" \
    JSONL_INPUT="${JSONL_INPUT}" \
    JSONL_MAX_EXAMPLES="${JSONL_MAX_EXAMPLES}" \
    TOKENIZER_JSON="${TOKENIZER_JSON}" \
    SKIP_OPTIMIZER_STEP=false \
      tools/distributed/run_qwen_pp_tp_ppo_trainer.sh "${MODEL_DIR}"
  ) | tee "${log}"

  if grep -Ei '(^|[^a-z])(nan|inf)([^a-z]|$)' "${log}" >/dev/null; then
    echo "${name} produced non-finite values" >&2
    exit 1
  fi
  if ! grep -q "step=$((STEPS - 1))" "${log}"; then
    echo "${name} did not reach step $((STEPS - 1))" >&2
    exit 1
  fi
  if ! grep -q "all_ranks_have_grad=true" "${log}"; then
    echo "${name} did not report gradients on every rank" >&2
    exit 1
  fi
  if ! grep -q "all_ranks_updated=true" "${log}"; then
    echo "${name} did not update every rank" >&2
    exit 1
  fi
}

require_file "${MODEL_DIR}" "model directory"
require_file "${JSONL_INPUT}" "GSM8K JSONL input"
require_file "${TOKENIZER_JSON}" "tokenizer JSON"
require_exec "${BUILD_DIR}/qwen3_5_pp_tp_ppo_trainer"

if [[ "${RUN_BASIC_TESTS}" = "1" ]]; then
  require_exec "${BUILD_DIR}/test_fp32_master_adamw"
  require_exec "${BUILD_DIR}/test_distributed_topology"
  require_exec "${BUILD_DIR}/test_nccl_collectives"
  "${BUILD_DIR}/test_fp32_master_adamw" | tee "${OUT_DIR}/test_fp32_master_adamw.log"
  "${BUILD_DIR}/test_distributed_topology" | tee "${OUT_DIR}/test_distributed_topology.log"
  (
    cd "${ROOT_DIR}"
    NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
    NCCL_DEBUG="${NCCL_DEBUG}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    WORLD_SIZE=4 \
      tools/distributed/run_nccl_smoke.sh "${BUILD_DIR}"
  ) | tee "${OUT_DIR}/test_nccl_collectives.log"
fi

run_case "qwen_pp2_tp2_ppo_5step" 2 2 2
run_case "qwen_pp4_tp1_ppo_5step" 4 1 4

echo "multi-GPU Qwen PPO regression passed"
