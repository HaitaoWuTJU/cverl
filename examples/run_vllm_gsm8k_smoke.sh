#!/usr/bin/env bash
# End-to-end vLLM + cverl GRPO smoke on the H20 box.
#
# Pipeline:
#   GSM8K JSONL -> vLLM HTTP rollout -> cverl C++ rule reward
#                                     -> GRPO advantage
#                                     -> Qwen3_5CausalLmPolicy fp32 PPO step
#
# Defaults run the FULL 24-layer Qwen3.5-0.8B (QWEN_MAX_LAYERS=-1) and verify
# that every layer's parameters move under PPO+KL.
#
# Quick run:
#
#   VLLM_PYTHON=/home/$(id -un)/vllm-env/bin/python \
#   BUILD_DIR=/home/$(id -un)/cverl-build/gpu-nccl \
#   MODEL_PATH=/path/to/Qwen3.5-0.8B \
#   examples/run_vllm_gsm8k_smoke.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-h20-nccl}"
DATASET="${DATASET:-${ROOT_DIR}/data/gsm8k-train-smoke.jsonl}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/../models/Qwen3.5-0.8B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8021}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"

# Trainer-side defaults — kept conservative because 24-layer Qwen3.5 fp32
# forward+backward through the linear-attention per-token loop is memory
# heavy (the 256-step Python loop materializes one (B, vh, kd, vd) tensor
# per token and keeps them all alive for autograd). On a single H20 with
# vLLM holding ~43 GiB on GPU 0, B=4 with ctx=384 fits; B=8 OOMs.
PROMPTS="${PROMPTS:-1}"
N="${N:-4}"
STEPS="${STEPS:-8}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-128}"
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-256}"
LR="${LR:-3e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
KL_COEF="${KL_COEF:-0.05}"
KL_PENALTY="${KL_PENALTY:-k2}"
TEMPERATURE="${TEMPERATURE:-1.1}"
TOP_P="${TOP_P:-0.95}"
REWARD_METHOD="${REWARD_METHOD:-flexible}"
ENDPOINT="${ENDPOINT:-chat}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a math tutor. Solve the problem step by step, then on the very last line write '#### N' where N is the final numeric answer with no units.}"
POLICY="${POLICY:-qwen}"
QWEN_MAX_LAYERS="${QWEN_MAX_LAYERS:--1}"
TRAINER_DEVICE="${TRAINER_DEVICE:-cuda}"
EXPORT_DIR="${EXPORT_DIR:-}"
EXPORT_EVERY="${EXPORT_EVERY:-1}"
EXPORT_DTYPE="${EXPORT_DTYPE:-bfloat16}"
RELOAD_URL="${RELOAD_URL:-}"
RELOAD_API_KEY="${RELOAD_API_KEY:-}"

# vLLM lives in its own Python (torch 2.11.0+cu128 from pytorch.org via the
# proxy + the 0.22.0+cu129 wheel from wheels.vllm.ai). Override via env.
VLLM_PYTHON="${VLLM_PYTHON:-/home/marvinhtwu/vllm-env/bin/python}"

# Trainer-side LD_LIBRARY_PATH: point at the conda env that built libcverl.a.
TRAINER_LIBS="${TRAINER_LIBS:-/apdcephfs_fsgm3/share_305110755/hunyuan/marvinhtwu/miniconda3/lib:/usr/local/cuda/lib64}"

# GPU split: vLLM on GPU 0, trainer on GPU 1 by default.
VLLM_DEVICES="${VLLM_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
TRAINER_DEVICES="${TRAINER_DEVICES:-1}"

if ! "${VLLM_PYTHON}" -c "import vllm" >/dev/null 2>&1; then
  echo "vllm not importable in ${VLLM_PYTHON}" >&2
  exit 1
fi

if [[ ! -x "${BUILD_DIR}/gsm8k_grpo_smoke" ]]; then
  echo "missing ${BUILD_DIR}/gsm8k_grpo_smoke; build cverl first" >&2
  exit 1
fi

LOG_FILE="${LOG_FILE:-/tmp/cverl_vllm_${UID}.log}"
PID_FILE="${PID_FILE:-/tmp/cverl_vllm_${UID}.pid}"
MODELS_FILE="${MODELS_FILE:-/tmp/cverl_vllm_models_${UID}.json}"
PROBE_FILE="${PROBE_FILE:-/tmp/cverl_vllm_completion_${UID}.json}"
rm -f "${LOG_FILE}" "${PID_FILE}" "${MODELS_FILE}" "${PROBE_FILE}" 2>/dev/null || true

cleanup() {
  if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    kill "$(cat "${PID_FILE}")" 2>/dev/null || true
  fi
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES="${VLLM_DEVICES}" \
"${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype "${DTYPE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  >"${LOG_FILE}" 2>&1 &
echo $! > "${PID_FILE}"

for _ in $(seq 1 120); do
  if curl -sf "http://${HOST}:${PORT}/v1/models" -o "${MODELS_FILE}"; then
    break
  fi
  if ! kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    tail -200 "${LOG_FILE}" >&2 || true
    exit 1
  fi
  sleep 5
done

# Sanity probe.
curl -sf "http://${HOST}:${PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL_PATH}\",\"prompt\":\"Question: What is 1+1? Answer with #### final number.\\n\",\"max_tokens\":32,\"temperature\":0,\"n\":1}" \
  >"${PROBE_FILE}"

TRAINER_EXTRA_ARGS=()
if [[ -n "${EXPORT_DIR}" ]]; then
  TRAINER_EXTRA_ARGS+=(--export-dir "${EXPORT_DIR}" --export-every "${EXPORT_EVERY}" --export-dtype "${EXPORT_DTYPE}")
fi
if [[ -n "${RELOAD_URL}" ]]; then
  TRAINER_EXTRA_ARGS+=(--reload-url "${RELOAD_URL}")
fi
if [[ -n "${RELOAD_API_KEY}" ]]; then
  TRAINER_EXTRA_ARGS+=(--reload-api-key "${RELOAD_API_KEY}")
fi

LD_LIBRARY_PATH="${TRAINER_LIBS}:${LD_LIBRARY_PATH:-}" \
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
CUDA_VISIBLE_DEVICES="${TRAINER_DEVICES}" \
"${BUILD_DIR}/gsm8k_grpo_smoke" \
  --dataset "${DATASET}" \
  --transport http \
  --base-url "http://${HOST}:${PORT}" \
  --endpoint "${ENDPOINT}" \
  --model "${MODEL_PATH}" \
  --system-prompt "${SYSTEM_PROMPT}" \
  --prompts "${PROMPTS}" \
  --n "${N}" \
  --steps "${STEPS}" \
  --ppo-epochs "${PPO_EPOCHS}" \
  --max-tokens "${MAX_TOKENS}" \
  --max-prompt-tokens "${MAX_PROMPT_TOKENS}" \
  --max-response-tokens "${MAX_RESPONSE_TOKENS}" \
  --tokenizer hf \
  --tokenizer-path "${MODEL_PATH}/tokenizer.json" \
  --policy "${POLICY}" \
  --model-dir "${MODEL_PATH}" \
  --qwen-max-layers "${QWEN_MAX_LAYERS}" \
  --device "${TRAINER_DEVICE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --lr "${LR}" \
  --reward-method "${REWARD_METHOD}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --kl-coef "${KL_COEF}" \
  --kl-penalty "${KL_PENALTY}" \
  "${TRAINER_EXTRA_ARGS[@]}"
