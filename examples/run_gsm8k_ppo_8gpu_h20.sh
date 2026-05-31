#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-h20-ppo-8gpu}"
PYTHON="${PYTHON:-python}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/../models/Qwen3.5-0.8B}"
TOKENIZER_JSON="${TOKENIZER_JSON:-${MODEL_DIR}/tokenizer.json}"
DATASET="${DATASET:-${ROOT_DIR}/data/gsm8k-train-bench.jsonl}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${ROOT_DIR}/build/gsm8k-ppo-8gpu-rollouts}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BASE_URL="${VLLM_BASE_URL:-http://${HOST}:${PORT}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-ppo-8gpu}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-oracle}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

DP_SIZE="${DP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-4}"
CP_SIZE="${CP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-2}"
LAYERS="${LAYERS:-24}"
PROMPT_LEN="${PROMPT_LEN:-128}"
RESPONSE_LEN="${RESPONSE_LEN:-64}"
MICRO_BATCHES="${MICRO_BATCHES:-${PP_SIZE}}"
STEPS="${STEPS:-5}"
PROMPTS="${PROMPTS:-2}"
N="${N:-2}"
MAX_TOKENS="${MAX_TOKENS:-64}"
LR="${LR:-1e-10}"
DTYPE="${DTYPE:-bfloat16}"
MEASURE_PARAM_DELTA="${MEASURE_PARAM_DELTA:-false}"
DP_GRAD_COMM_DTYPE="${DP_GRAD_COMM_DTYPE:-model}"
TP_GRAD_COMM_DTYPE="${TP_GRAD_COMM_DTYPE:-${DP_GRAD_COMM_DTYPE}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/build/gsm8k-ppo-8gpu-checkpoints}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-0}"
METRICS_CSV="${METRICS_CSV:-${ROOT_DIR}/build/metrics/gsm8k_ppo_8gpu.csv}"
ID_PREFIX="${ID_PREFIX:-/tmp/cverl_gsm8k_ppo_8gpu}"

CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-$("${PYTHON}" - <<'PY'
import torch
print(torch.utils.cmake_prefix_path)
PY
)}"
TORCH_LIB_DIR="$("${PYTHON}" - <<'PY'
import pathlib
import torch
print(pathlib.Path(torch.__file__).resolve().parent / "lib")
PY
)"

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]] && ! grep -q "${TORCH_LIB_DIR}/libc10.so" "${BUILD_DIR}/CMakeCache.txt"; then
  echo "Removing stale build dir with mismatched Torch cache: ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi

mkdir -p "${ROLLOUT_DIR}" "$(dirname "${METRICS_CSV}")"
rm -f "${ROLLOUT_DIR}"/rollout_step_*.json

CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCVERL_ENABLE_CUDA=ON \
  -DCVERL_ENABLE_NCCL=ON \
  -DCVERL_BUILD_PYTHON_BRIDGE=ON
cmake --build "${BUILD_DIR}" -j "${JOBS:-8}" --target qwen3_5_pp_tp_ppo_trainer _cverl_vllm_bridge

PYTHONPATH="${BUILD_DIR}:${PYTHONPATH:-}" \
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
"${PYTHON}" "${ROOT_DIR}/tools/rollout/vllm_online_grpo.py" \
  --dataset "${DATASET}" \
  --model-dir "${MODEL_DIR}" \
  --tokenizer-path "${TOKENIZER_JSON}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --base-url "${BASE_URL}" \
  --rollout-backend "${ROLLOUT_BACKEND}" \
  --skip-train \
  --dump-rollout-dir "${ROLLOUT_DIR}" \
  --steps "${STEPS}" \
  --prompts "${PROMPTS}" \
  --n "${N}" \
  --max-tokens "${MAX_TOKENS}" \
  --max-prompt-tokens "${PROMPT_LEN}" \
  --max-response-tokens "${RESPONSE_LEN}"

(
  cd "${ROOT_DIR}"
  NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
  NCCL_DEBUG="${NCCL_DEBUG}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  BUILD_DIR="${BUILD_DIR}" \
  DP_SIZE="${DP_SIZE}" \
  PP_SIZE="${PP_SIZE}" \
  CP_SIZE="${CP_SIZE}" \
  TP_SIZE="${TP_SIZE}" \
  LOCAL_WORLD_SIZE="$((DP_SIZE * PP_SIZE * CP_SIZE * TP_SIZE))" \
  LAYERS="${LAYERS}" \
  PROMPT_LEN="${PROMPT_LEN}" \
  RESPONSE_LEN="${RESPONSE_LEN}" \
  MICRO_BATCHES="${MICRO_BATCHES}" \
  STEPS="${STEPS}" \
  LR="${LR}" \
  DTYPE="${DTYPE}" \
  MEASURE_PARAM_DELTA="${MEASURE_PARAM_DELTA}" \
  DP_GRAD_COMM_DTYPE="${DP_GRAD_COMM_DTYPE}" \
  TP_GRAD_COMM_DTYPE="${TP_GRAD_COMM_DTYPE}" \
  ROLLOUT_DIR="${ROLLOUT_DIR}" \
  TOKENIZER_JSON="${TOKENIZER_JSON}" \
  CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
  CHECKPOINT_EVERY="${CHECKPOINT_EVERY}" \
  METRICS_CSV="${METRICS_CSV}" \
  ID_PREFIX="${ID_PREFIX}" \
    tools/distributed/run_qwen_pp_tp_ppo_trainer.sh "${MODEL_DIR}"
)

echo "rollout_dir=${ROLLOUT_DIR}"
echo "metrics_csv=${METRICS_CSV}"
