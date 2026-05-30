#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-h20-nccl}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/../models/Qwen3.5-0.8B}"
TOKENIZER_JSON="${TOKENIZER_JSON:-${MODEL_DIR}/tokenizer.json}"
DATASET="${DATASET:-${ROOT_DIR}/data/gsm8k-train-bench.jsonl}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${ROOT_DIR}/build/static-split-rollouts}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BASE_URL="${VLLM_BASE_URL:-http://${HOST}:${PORT}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-static-split}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth1}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"

# 4-card test layout for the 8-card production design:
#   GPU0-1: rollout service, vLLM TP=2
#   GPU2-3: trainer, PP/TP world=2
VLLM_GPUS="${VLLM_GPUS:-0,1}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-2}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
TRAINER_GPUS="${TRAINER_GPUS:-2,3}"
PP_SIZE="${PP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-1}"

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
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/build/static-split-checkpoints}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-0}"
METRICS_CSV="${METRICS_CSV:-${ROOT_DIR}/build/metrics/static_split_4gpu_pp_train.csv}"
VLLM_LOG="${VLLM_LOG:-/tmp/cverl_vllm_static_split.log}"

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

mkdir -p "${ROLLOUT_DIR}" "$(dirname "${METRICS_CSV}")"
rm -f "${ROLLOUT_DIR}"/rollout_step_*.json

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCVERL_ENABLE_CUDA=ON \
  -DCVERL_ENABLE_NCCL=ON \
  -DCVERL_BUILD_PYTHON_BRIDGE=ON
cmake --build "${BUILD_DIR}" -j "${JOBS:-8}" --target qwen3_5_pp_tp_ppo_trainer _cverl_vllm_bridge

if [[ "${ROLLOUT_BACKEND}" == "vllm" ]]; then
  VLLM_MAX_MODEL_LEN_ARGS=()
  if [[ -n "${VLLM_MAX_MODEL_LEN}" ]]; then
    VLLM_MAX_MODEL_LEN_ARGS=(--max-model-len "${VLLM_MAX_MODEL_LEN}")
  fi

  VLLM_SERVER_DEV_MODE=1 \
  NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
  CUDA_VISIBLE_DEVICES="${VLLM_GPUS}" \
  nohup vllm serve "${MODEL_DIR}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype "${VLLM_DTYPE:-bfloat16}" \
    --tensor-parallel-size "${VLLM_TP_SIZE}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    "${VLLM_MAX_MODEL_LEN_ARGS[@]}" \
    --enable-sleep-mode \
    --weight-transfer-config '{"backend":"nccl"}' \
    > "${VLLM_LOG}" 2>&1 &
  VLLM_PID=$!

  for i in $(seq 1 "${VLLM_READY_RETRIES:-180}"); do
    if curl -fsS "${BASE_URL}/v1/models" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      tail -160 "${VLLM_LOG}" || true
      exit 1
    fi
    sleep 2
  done
fi

PYTHONPATH="${BUILD_DIR}:${PYTHONPATH:-}" \
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
"${ROOT_DIR}/tools/rollout/vllm_online_grpo.py" \
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
  NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
  CUDA_VISIBLE_DEVICES="${TRAINER_GPUS}" \
  BUILD_DIR="${BUILD_DIR}" \
  PP_SIZE="${PP_SIZE}" \
  TP_SIZE="${TP_SIZE}" \
  LAYERS="${LAYERS}" \
  PROMPT_LEN="${PROMPT_LEN}" \
  RESPONSE_LEN="${RESPONSE_LEN}" \
  MICRO_BATCHES="${MICRO_BATCHES}" \
  STEPS="${STEPS}" \
  LR="${LR}" \
  DTYPE="${DTYPE}" \
  ROLLOUT_DIR="${ROLLOUT_DIR}" \
  TOKENIZER_JSON="${TOKENIZER_JSON}" \
  CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
  CHECKPOINT_EVERY="${CHECKPOINT_EVERY}" \
  METRICS_CSV="${METRICS_CSV}" \
    tools/distributed/run_qwen_pp_tp_ppo_trainer.sh "${MODEL_DIR}"
)

echo "rollout_dir=${ROLLOUT_DIR}"
echo "metrics_csv=${METRICS_CSV}"
