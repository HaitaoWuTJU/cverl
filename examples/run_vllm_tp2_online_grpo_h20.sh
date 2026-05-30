#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-h20-nccl}"
DATASET="${DATASET:-${ROOT_DIR}/data/gsm8k-train-smoke.jsonl}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/../models/Qwen3.5-0.8B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_DIR}/tokenizer.json}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
BASE_URL="${VLLM_BASE_URL:-http://${HOST}:${PORT}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-tp2}"
VLLM_GPUS="${VLLM_GPUS:-0,1}"
TRAINER_GPUS="${TRAINER_GPUS:-2}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth1}"
MASTER_PORT="${MASTER_PORT:-29601}"

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCVERL_ENABLE_NCCL=ON \
  -DCVERL_BUILD_PYTHON_BRIDGE=ON
cmake --build "${BUILD_DIR}" -j "${JOBS:-2}" --target _cverl_vllm_bridge

VLLM_SERVER_DEV_MODE=1 \
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
CUDA_VISIBLE_DEVICES="${VLLM_GPUS}" \
nohup vllm serve "${MODEL_DIR}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype "${VLLM_DTYPE:-bfloat16}" \
  --tensor-parallel-size "${VLLM_TP_SIZE:-2}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.70}" \
  --enable-sleep-mode \
  --weight-transfer-config '{"backend":"nccl"}' \
  > "${VLLM_LOG:-/tmp/cverl_vllm_tp2.log}" 2>&1 &
VLLM_PID=$!

for i in $(seq 1 "${VLLM_READY_RETRIES:-120}"); do
  if curl -fsS "${BASE_URL}/v1/models" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    tail -120 "${VLLM_LOG:-/tmp/cverl_vllm_tp2.log}" || true
    exit 1
  fi
  sleep 2
  if [[ $((i % 10)) -eq 0 ]]; then
    tail -20 "${VLLM_LOG:-/tmp/cverl_vllm_tp2.log}" || true
  fi
done

VLLM_WORLD_SIZE="$(python - "${BASE_URL}" <<'PY'
import sys
import requests

doc = requests.get(sys.argv[1].rstrip("/") + "/get_world_size?include_dp=true", timeout=30).json()
print(doc["world_size"])
PY
)"
WORLD_SIZE="${WORLD_SIZE:-$((VLLM_WORLD_SIZE + 1))}"

args=(
  --dataset "${DATASET}"
  --model-dir "${MODEL_DIR}"
  --tokenizer-path "${TOKENIZER_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --base-url "${BASE_URL}"
  --device "${DEVICE:-cuda:0}"
  --param-dtype "${PARAM_DTYPE:-bfloat16}"
  --qwen-max-layers "${QWEN_MAX_LAYERS:--1}"
  --rollout-backend "${ROLLOUT_BACKEND:-vllm}"
  --steps "${STEPS:-1}"
  --prompts "${PROMPTS:-1}"
  --n "${N:-2}"
  --max-tokens "${MAX_TOKENS:-16}"
  --max-prompt-tokens "${MAX_PROMPT_TOKENS:-64}"
  --max-response-tokens "${MAX_RESPONSE_TOKENS:-64}"
  --ppo-epochs "${PPO_EPOCHS:-1}"
  --master-address "${MASTER_ADDRESS:-127.0.0.1}"
  --master-port "${MASTER_PORT}"
  --world-size "${WORLD_SIZE}"
  --sync-every "${SYNC_EVERY:-1}"
)

if [[ "${MEASURE_PARAM_DELTA:-0}" == "1" ]]; then
  args+=(--measure-param-delta)
fi
if [[ "${ROLLOUT_BACKEND:-vllm}" == "oracle" ]]; then
  args+=(--measure-param-delta)
fi

PYTHONPATH="${BUILD_DIR}:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES="${TRAINER_GPUS}" \
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
"${ROOT_DIR}/tools/rollout/vllm_online_grpo.py" "${args[@]}"
