#!/usr/bin/env bash
# End-to-end SGLang + cverl GRPO trainer on the H20 box.
#
# Pipeline:
#   GSM8K JSONL -> SGLang HTTP rollout -> cverl C++ rule reward
#                                       -> GRPO advantage
#                                       -> Qwen3_5CausalLmPolicy fp32 PPO step
#
# The trainer side ("cverl gsm8k_grpo_trainer") talks to a stock SGLang server
# over /v1/chat/completions; both ends share the same Qwen3.5 weights so the
# PPO update is meaningful. SGLang lives on GPU 0 by default, the trainer on
# GPU 1, so they don't fight for memory.
#
# Quick run with the H20 defaults (sglang lives in /home/marvinhtwu/sglang-env
# in our setup):
#
#   SGLANG_PYTHON=/home/marvinhtwu/sglang-env/bin/python \
#   BUILD_DIR=/home/marvinhtwu/cverl-build/gpu-nccl \
#   MODEL_PATH=/path/to/Qwen3.5-0.8B \
#   examples/run_sglang_gsm8k_train.sh
#
# Required runtime extras for SGLang 0.5.9 on this node:
#   pip install --no-cache-dir IPython decord2 nvidia-cudnn-cu12==9.16.0.29 \
#                              flashinfer_python==0.6.3 flashinfer_cubin==0.6.3
# (the cuDNN upgrade silences the 9.10 vs Conv3D-bug check; we set
# SGLANG_DISABLE_CUDNN_CHECK=1 below regardless.)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-h20-nccl}"
DATASET="${DATASET:-${ROOT_DIR}/data/gsm8k-train-bench.jsonl}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/../models/Qwen3.5-0.8B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8011}"
DTYPE="${DTYPE:-bfloat16}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.45}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1024}"
PROMPTS="${PROMPTS:-4}"
N="${N:-4}"
STEPS="${STEPS:-1}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-256}"
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-256}"
LR="${LR:-3e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
KL_COEF="${KL_COEF:-0.0}"
KL_PENALTY="${KL_PENALTY:-k2}"
TEMPERATURE="${TEMPERATURE:-1.1}"
TOP_P="${TOP_P:-0.95}"
REWARD_METHOD="${REWARD_METHOD:-flexible}"
ENDPOINT="${ENDPOINT:-chat}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a math tutor. Solve the problem step by step, then on the very last line write '#### N' where N is the final numeric answer with no units.}"
POLICY="${POLICY:-qwen}"
QWEN_MAX_LAYERS="${QWEN_MAX_LAYERS:-2}"
TRAINER_DEVICE="${TRAINER_DEVICE:-cuda}"
EXPORT_DIR="${EXPORT_DIR:-}"
EXPORT_EVERY="${EXPORT_EVERY:-1}"
EXPORT_DTYPE="${EXPORT_DTYPE:-bfloat16}"

# SGLang server runs in its own Python (torch 2.9.1 + sgl_kernel 0.3.21).
# Default to the env we built under /home/marvinhtwu; override SGLANG_PYTHON.
SGLANG_PYTHON="${SGLANG_PYTHON:-/home/marvinhtwu/sglang-env/bin/python}"

# Trainer-side LD_LIBRARY_PATH: point at the conda env that built libcverl.a
# so the right libtorch/cuda runtime gets loaded next to gsm8k_grpo_trainer.
TRAINER_LIBS="${TRAINER_LIBS:-/apdcephfs_fsgm3/share_305110755/hunyuan/marvinhtwu/miniconda3/lib:/usr/local/cuda/lib64}"

# GPU split: SGLang on GPU 0, trainer on GPU 1 by default.
SGLANG_DEVICES="${SGLANG_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
TRAINER_DEVICES="${TRAINER_DEVICES:-1}"

if ! "${SGLANG_PYTHON}" -c "import sglang" >/dev/null 2>&1; then
  echo "sglang not importable in ${SGLANG_PYTHON}" >&2
  exit 1
fi

if [[ ! -x "${BUILD_DIR}/gsm8k_grpo_trainer" ]]; then
  echo "missing ${BUILD_DIR}/gsm8k_grpo_trainer; build cverl first" >&2
  exit 1
fi

LOG_FILE="${LOG_FILE:-/tmp/cverl_sglang_${UID}.log}"
PID_FILE="${PID_FILE:-/tmp/cverl_sglang_${UID}.pid}"
MODELS_FILE="${MODELS_FILE:-/tmp/cverl_sglang_models_${UID}.json}"
PROBE_FILE="${PROBE_FILE:-/tmp/cverl_sglang_completion_${UID}.json}"
rm -f "${LOG_FILE}" "${PID_FILE}" "${MODELS_FILE}" "${PROBE_FILE}" 2>/dev/null || true

cleanup() {
  if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    kill "$(cat "${PID_FILE}")" 2>/dev/null || true
  fi
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES="${SGLANG_DEVICES}" \
SGLANG_DISABLE_CUDNN_CHECK=1 \
"${SGLANG_PYTHON}" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --context-length "${CONTEXT_LENGTH}" \
  --dtype "${DTYPE}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --disable-cuda-graph \
  --sampling-backend pytorch \
  --mamba-backend flashinfer \
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
LD_LIBRARY_PATH="${TRAINER_LIBS}:${LD_LIBRARY_PATH:-}" \
CUDA_VISIBLE_DEVICES="${TRAINER_DEVICES}" \
"${BUILD_DIR}/gsm8k_grpo_trainer" \
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
