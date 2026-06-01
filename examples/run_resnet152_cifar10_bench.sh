#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-h20-ppo-8gpu-zw}"
PYTHON="${PYTHON:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-cuda:0}"

CIFAR_ROOT="${CIFAR_ROOT:-/data/workspace/ceph/zw/rl/datasets/cifar10}"
CIFAR_DIR="${CIFAR_DIR:-${CIFAR_ROOT}/hf_tensors}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
WARMUP_BATCHES="${WARMUP_BATCHES:-10}"
SEED="${SEED:-1234}"
LR="${LR:-0.1}"
MOMENTUM="${MOMENTUM:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
ALLOW_TF32="${ALLOW_TF32:-true}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/build/bench/resnet152_cifar10}"
RUN_CPP="${RUN_CPP:-1}"
RUN_PYTHON="${RUN_PYTHON:-1}"
PREPARE_DATA="${PREPARE_DATA:-0}"

mkdir -p "${CIFAR_ROOT}" "${OUT_DIR}"

if [[ ! -f "${CIFAR_DIR}/train_images.f32" || ! -f "${CIFAR_DIR}/train_labels.i64" || ! -f "${CIFAR_DIR}/test_images.f32" || ! -f "${CIFAR_DIR}/test_labels.i64" ]]; then
  if [[ "${PREPARE_DATA}" == "1" ]]; then
    DATA_ROOT="${CIFAR_ROOT}" TENSOR_DIR="${CIFAR_DIR}" "${ROOT_DIR}/examples/prepare_cifar10_hf.sh"
  else
    cat >&2 <<EOF
Missing CIFAR10 tensor cache:
  ${CIFAR_DIR}/train_images.f32
  ${CIFAR_DIR}/train_labels.i64
  ${CIFAR_DIR}/test_images.f32
  ${CIFAR_DIR}/test_labels.i64

Prepare it on the local/shared filesystem first:
  DATA_ROOT=${CIFAR_ROOT} TENSOR_DIR=${CIFAR_DIR} examples/prepare_cifar10_hf.sh

Set PREPARE_DATA=1 only when running on a machine where downloads are allowed.
EOF
    exit 1
  fi
fi

cmake --build "${BUILD_DIR}" --target resnet152_cifar10_cpp_bench -j "${JOBS:-8}"

common_args=(
  --data-dir "${CIFAR_DIR}"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --warmup-batches "${WARMUP_BATCHES}"
  --seed "${SEED}"
  --lr "${LR}"
  --momentum "${MOMENTUM}"
  --weight-decay "${WEIGHT_DECAY}"
  --allow-tf32 "${ALLOW_TF32}"
)

cpp_json="${OUT_DIR}/cpp_summary.json"
cpp_csv="${OUT_DIR}/cpp_epochs.csv"
py_json="${OUT_DIR}/python_summary.json"
py_csv="${OUT_DIR}/python_epochs.csv"
compare_csv="${OUT_DIR}/comparison.csv"

wall_now() {
  "${PYTHON}" - <<'PY'
import time
print(f"{time.perf_counter():.9f}")
PY
}

annotate_wall_json() {
  local json_path="$1"
  local wall_seconds="$2"
  "${PYTHON}" - "${json_path}" "${wall_seconds}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
wall_seconds = float(sys.argv[2])
if path.exists():
    row = json.loads(path.read_text())
else:
    row = {}
row["wall_seconds"] = wall_seconds
row["comparison_total_seconds"] = wall_seconds
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(row, indent=2) + "\n")
PY
}

run_backend() {
  local json_path="$1"
  local log_path="$2"
  shift 2
  local begin end wall status
  begin="$(wall_now)"
  set +e
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "$@" 2>&1 | tee "${log_path}"
  status=${PIPESTATUS[0]}
  set -e
  end="$(wall_now)"
  wall="$("${PYTHON}" - "${begin}" "${end}" <<'PY'
import sys
print(f"{float(sys.argv[2]) - float(sys.argv[1]):.9f}")
PY
)"
  annotate_wall_json "${json_path}" "${wall}"
  return "${status}"
}

if [[ "${RUN_CPP}" == "1" ]]; then
  run_backend "${cpp_json}" "${OUT_DIR}/cpp.log" \
    "${BUILD_DIR}/resnet152_cifar10_cpp_bench" \
      "${common_args[@]}" \
      --metrics-csv "${cpp_csv}" \
      --summary-json "${cpp_json}"
fi

if [[ "${RUN_PYTHON}" == "1" ]]; then
  run_backend "${py_json}" "${OUT_DIR}/python.log" \
    "${PYTHON}" tools/bench/resnet152_cifar10_python_bench.py \
      "${common_args[@]}" \
      --metrics-csv "${py_csv}" \
      --summary-json "${py_json}"
fi

"${PYTHON}" - "${cpp_json}" "${py_json}" "${compare_csv}" <<'PY'
import csv
import json
import sys
from pathlib import Path

cpp_path = Path(sys.argv[1])
py_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])
rows = []
if cpp_path.exists():
    rows.append(json.loads(cpp_path.read_text()))
if py_path.exists():
    rows.append(json.loads(py_path.read_text()))
def cmp_seconds(row):
    return float(row.get("comparison_total_seconds", row.get("wall_seconds", row.get("total_seconds", float("nan")))))

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "backend",
            "epochs",
            "batch_size",
            "comparison_total_seconds",
            "wall_seconds",
            "end_to_end_seconds",
            "epoch_train_seconds",
            "seconds_per_epoch",
            "data_seconds",
            "model_init_seconds",
            "warmup_seconds",
            "eval_seconds",
            "examples_per_second",
            "final_loss",
            "train_loss",
            "train_accuracy",
            "test_loss",
            "test_accuracy",
        ],
    )
    writer.writeheader()
    for row in rows:
        examples = row["examples"] * row["epochs"]
        total = cmp_seconds(row)
        writer.writerow(
            {
                "backend": row["backend"],
                "epochs": row["epochs"],
                "batch_size": row["batch_size"],
                "comparison_total_seconds": f'{total:.6f}',
                "wall_seconds": f'{row.get("wall_seconds", float("nan")):.6f}',
                "end_to_end_seconds": f'{row.get("end_to_end_seconds", row.get("total_seconds", float("nan"))):.6f}',
                "epoch_train_seconds": f'{row.get("epoch_train_seconds", float("nan")):.6f}',
                "seconds_per_epoch": f'{row["seconds_per_epoch"]:.6f}',
                "data_seconds": f'{row.get("data_seconds", float("nan")):.6f}',
                "model_init_seconds": f'{row.get("model_init_seconds", float("nan")):.6f}',
                "warmup_seconds": f'{row.get("warmup_seconds", float("nan")):.6f}',
                "eval_seconds": f'{row.get("eval_seconds", float("nan")):.6f}',
                "examples_per_second": f'{examples / total:.3f}',
                "final_loss": f'{row["final_loss"]:.6f}',
                "train_loss": f'{row.get("train_loss", float("nan")):.6f}',
                "train_accuracy": f'{row.get("train_accuracy", float("nan")):.6f}',
                "test_loss": f'{row.get("test_loss", float("nan")):.6f}',
                "test_accuracy": f'{row.get("test_accuracy", float("nan")):.6f}',
            }
        )

by_backend = {row["backend"]: row for row in rows}
for row in rows:
    print(
        f'{row["backend"]}_metrics '
        f'train_loss={row.get("train_loss", float("nan")):.6f} '
        f'train_acc={row.get("train_accuracy", float("nan")):.6f} '
        f'test_loss={row.get("test_loss", float("nan")):.6f} '
        f'test_acc={row.get("test_accuracy", float("nan")):.6f}'
)
if "cpp_libtorch" in by_backend and "python_pytorch" in by_backend:
    cpp = cmp_seconds(by_backend["cpp_libtorch"])
    py = cmp_seconds(by_backend["python_pytorch"])
    speedup = py / cpp if cpp > 0 else float("nan")
    print(f"comparison_csv={out_path}")
    print(f"cpp_total_seconds={cpp:.6f}")
    print(f"python_total_seconds={py:.6f}")
    print(f"cpp_speedup_vs_python={speedup:.4f}x")
else:
    print(f"comparison_csv={out_path}")
PY
