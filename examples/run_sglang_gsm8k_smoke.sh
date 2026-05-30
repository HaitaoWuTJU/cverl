#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-h20-nccl}"
DATASET="${DATASET:-${ROOT_DIR}/data/gsm8k-train-smoke.jsonl}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/../models/Qwen3.5-0.8B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
DTYPE="${DTYPE:-bfloat16}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.45}"
PROMPTS="${PROMPTS:-4}"
N="${N:-4}"
MAX_TOKENS="${MAX_TOKENS:-256}"
PATCH_SGLANG_ACTIVATION_JIT="${PATCH_SGLANG_ACTIVATION_JIT:-0}"

if ! python -c "import sglang" >/dev/null 2>&1; then
  echo "sglang is not importable in the active Python environment" >&2
  exit 1
fi

if [[ ! -x "${BUILD_DIR}/gsm8k_grpo_smoke" ]]; then
  echo "missing ${BUILD_DIR}/gsm8k_grpo_smoke; build cverl first" >&2
  exit 1
fi

if [[ "${PATCH_SGLANG_ACTIVATION_JIT}" == "1" ]]; then
  python - <<'PY'
from pathlib import Path
import sglang

root = Path(sglang.__file__).resolve().parent
path = root / "jit_kernel/csrc/elementwise/activation.cuh"
text = path.read_text()
old = '''  template <ActivationKind kAct, bool kFilterExpert>
  static constexpr auto activation_kernel = act_and_mul_kernel<T, kAct, kUsePDL, kFilterExpert>;

  static_assert(device::kMaxVecBytes % sizeof(T) == 0, "unsupported data type");

  template <bool kFilterExpert>
  static auto select_kernel(const std::string& type)
      -> decltype(activation_kernel<ActivationKind::kSiLU, kFilterExpert>) {
    using namespace host;
    if (type == "silu") {
      return activation_kernel<ActivationKind::kSiLU, kFilterExpert>;
    } else if (type == "gelu") {
      return activation_kernel<ActivationKind::kGELU, kFilterExpert>;
    } else if (type == "gelu_tanh") {
      return activation_kernel<ActivationKind::kGELUTanh, kFilterExpert>;
    } else {
      Panic("unsupported activation type: ", type);
    }
    return nullptr;
  }
'''
new = '''  static_assert(device::kMaxVecBytes % sizeof(T) == 0, "unsupported data type");

  template <ActivationKind kAct, bool kFilterExpert>
  using KernelFn = decltype(&act_and_mul_kernel<T, kAct, kUsePDL, kFilterExpert>);

  template <bool kFilterExpert>
  static auto select_kernel(const std::string& type) -> KernelFn<ActivationKind::kSiLU, kFilterExpert> {
    using namespace host;
    if (type == "silu") {
      return &act_and_mul_kernel<T, ActivationKind::kSiLU, kUsePDL, kFilterExpert>;
    } else if (type == "gelu") {
      return &act_and_mul_kernel<T, ActivationKind::kGELU, kUsePDL, kFilterExpert>;
    } else if (type == "gelu_tanh") {
      return &act_and_mul_kernel<T, ActivationKind::kGELUTanh, kUsePDL, kFilterExpert>;
    } else {
      Panic("unsupported activation type: ", type);
    }
    return nullptr;
  }
'''
if old in text:
    path.write_text(text.replace(old, new))
    print(f"patched {path}")
else:
    print(f"activation JIT patch already applied or source changed: {path}")
PY
  rm -rf /root/.cache/tvm-ffi/sgl_kernel_jit_activation_* 2>/dev/null || true
fi

LOG_FILE="${LOG_FILE:-/tmp/cverl_sglang.log}"
PID_FILE="${PID_FILE:-/tmp/cverl_sglang.pid}"
rm -f "${LOG_FILE}" "${PID_FILE}"

cleanup() {
  if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    kill "$(cat "${PID_FILE}")" 2>/dev/null || true
  fi
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --context-length 2048 \
  --dtype "${DTYPE}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --disable-cuda-graph \
  --sampling-backend pytorch \
  --linear-attn-backend flashinfer \
  --mamba-backend flashinfer \
  >"${LOG_FILE}" 2>&1 &
echo $! > "${PID_FILE}"

for _ in $(seq 1 120); do
  if curl -sf "http://${HOST}:${PORT}/v1/models" >/tmp/cverl_sglang_models.json; then
    break
  fi
  if ! kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    tail -200 "${LOG_FILE}" >&2 || true
    exit 1
  fi
  sleep 5
done

curl -sf "http://${HOST}:${PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL_PATH}\",\"prompt\":\"Question: What is 1+1? Answer with #### final number.\\n\",\"max_tokens\":32,\"temperature\":0,\"n\":1}" \
  >/tmp/cverl_sglang_completion.json

"${BUILD_DIR}/gsm8k_grpo_smoke" \
  --dataset "${DATASET}" \
  --transport http \
  --base-url "http://${HOST}:${PORT}" \
  --endpoint completions \
  --model "${MODEL_PATH}" \
  --prompts "${PROMPTS}" \
  --n "${N}" \
  --steps 1 \
  --ppo-epochs 1 \
  --max-tokens "${MAX_TOKENS}" \
  --max-prompt-tokens 256 \
  --max-response-tokens 256 \
  --hidden-dim 32 \
  --tokenizer byte \
  --reward-method flexible \
  --temperature 0.9 \
  --top-p 0.95 \
  --kl-coef 0.01
