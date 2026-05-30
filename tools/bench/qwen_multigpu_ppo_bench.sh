#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-h20-nccl}"
MODEL_DIR="${1:-${MODEL_DIR:-${ROOT_DIR}/../models/Qwen3.5-0.8B}}"
JSONL_INPUT="${JSONL_INPUT:-${ROOT_DIR}/data/gsm8k-train-bench.jsonl}"
JSONL_MAX_EXAMPLES="${JSONL_MAX_EXAMPLES:-8}"
TOKENIZER_JSON="${TOKENIZER_JSON:-${MODEL_DIR}/tokenizer.json}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/build/bench}"
OUT_CSV="${OUT_CSV:-${OUT_DIR}/qwen_multigpu_ppo_bench.csv}"
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

mkdir -p "${OUT_DIR}"

if [[ ! -x "${BUILD_DIR}/qwen3_5_pp_tp_ppo_trainer" ]]; then
  cmake --build "${BUILD_DIR}" --target qwen3_5_pp_tp_ppo_trainer -j "${BUILD_JOBS:-8}"
fi

run_bench_case() {
  local name="$1"
  local pp="$2"
  local tp="$3"
  local micro_batches="$4"
  local log="${OUT_DIR}/${name}.bench.log"
  local start_ns
  local end_ns
  local seconds
  local tokens
  local steps_per_second
  local tokens_per_second

  start_ns="$(date +%s%N)"
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
  end_ns="$(date +%s%N)"

  if grep -Ei '(^|[^a-z])(nan|inf)([^a-z]|$)' "${log}" >/dev/null; then
    echo "${name} produced non-finite values" >&2
    exit 1
  fi
  if ! grep -q "step=$((STEPS - 1))" "${log}"; then
    echo "${name} did not reach step $((STEPS - 1))" >&2
    exit 1
  fi

  seconds="$(awk -v s="${start_ns}" -v e="${end_ns}" 'BEGIN { printf "%.6f", (e - s) / 1000000000.0 }')"
  tokens=$((STEPS * micro_batches * (PROMPT_LEN + RESPONSE_LEN)))
  steps_per_second="$(awk -v n="${STEPS}" -v s="${seconds}" 'BEGIN { printf "%.6f", n / s }')"
  tokens_per_second="$(awk -v n="${tokens}" -v s="${seconds}" 'BEGIN { printf "%.6f", n / s }')"
  echo "${name},${pp},${tp},${micro_batches},${STEPS},${LAYERS},${DTYPE},${seconds},${steps_per_second},${tokens_per_second},${log}" >> "${OUT_CSV}"
}

echo "case,pp,tp,micro_batches,steps,layers,dtype,seconds,steps_per_second,tokens_per_second,log" > "${OUT_CSV}"
run_bench_case "qwen_pp2_tp2_ppo" 2 2 2
run_bench_case "qwen_pp4_tp1_ppo" 4 1 4

cat "${OUT_CSV}"
