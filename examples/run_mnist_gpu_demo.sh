#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-h20-mnist}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/mnist/raw}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-512}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_TEST_BATCHES="${MAX_TEST_BATCHES:-0}"
LR="${LR:-1e-3}"
STOP_WHEN_TRAIN_ACC_GT_TEST="${STOP_WHEN_TRAIN_ACC_GT_TEST:-false}"
METRICS_CSV="${METRICS_CSV:-${ROOT_DIR}/build/metrics/mnist_gpu_demo.csv}"
CURVE_SVG="${CURVE_SVG:-${ROOT_DIR}/build/metrics/mnist_gpu_demo_loss_curve.svg}"

download_one() {
  local name="$1"
  local raw="${DATA_DIR}/${name}"
  local gz="${raw}.gz"
  if [[ -s "${raw}" ]]; then
    return
  fi
  mkdir -p "${DATA_DIR}"
  if [[ ! -s "${gz}" ]]; then
    local url
    local ok=0
    for url in \
      "https://storage.googleapis.com/cvdf-datasets/mnist/${name}.gz" \
      "https://ossci-datasets.s3.amazonaws.com/mnist/${name}.gz"; do
      echo "Downloading ${url}"
      if command -v curl >/dev/null 2>&1; then
        if curl -fL --retry 3 --connect-timeout 20 -o "${gz}.tmp" "${url}"; then
          mv "${gz}.tmp" "${gz}"
          ok=1
          break
        fi
      else
        if "${PYTHON}" -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' \
          "${url}" "${gz}.tmp"; then
          mv "${gz}.tmp" "${gz}"
          ok=1
          break
        fi
      fi
      rm -f "${gz}.tmp"
    done
    if [[ "${ok}" != "1" ]]; then
      echo "Failed to download ${name}.gz" >&2
      exit 1
    fi
  fi
  gzip -dc "${gz}" >"${raw}"
}

download_one train-images-idx3-ubyte
download_one train-labels-idx1-ubyte
download_one t10k-images-idx3-ubyte
download_one t10k-labels-idx1-ubyte

CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-$("${PYTHON}" -c 'import torch; print(torch.utils.cmake_prefix_path)')}"
TORCH_LIB_DIR="$("${PYTHON}" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")')"
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]] && ! grep -q "${TORCH_LIB_DIR}/libc10.so" "${BUILD_DIR}/CMakeCache.txt"; then
  echo "Removing stale build dir with mismatched Torch cache: ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi

mkdir -p "$(dirname "${METRICS_CSV}")" "$(dirname "${CURVE_SVG}")"
CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCVERL_ENABLE_CUDA=ON \
  -DCVERL_ENABLE_NCCL=OFF \
  -DCVERL_BUILD_PYTHON_BRIDGE=OFF
cmake --build "${BUILD_DIR}" -j "${JOBS:-8}" --target mnist_gpu_demo

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
EXTRA_ARGS=()
if [[ "${STOP_WHEN_TRAIN_ACC_GT_TEST}" == "true" ]]; then
  EXTRA_ARGS+=(--stop-when-train-acc-gt-test)
fi

"${BUILD_DIR}/mnist_gpu_demo" \
  --data-dir "${DATA_DIR}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --max-train-batches "${MAX_TRAIN_BATCHES}" \
  --max-test-batches "${MAX_TEST_BATCHES}" \
  --lr "${LR}" \
  --metrics-csv "${METRICS_CSV}" \
  "${EXTRA_ARGS[@]}"

"${PYTHON}" - "${METRICS_CSV}" "${CURVE_SVG}" <<'PY'
import csv
import math
import sys

csv_path, svg_path = sys.argv[1], sys.argv[2]
steps = []
losses = []
epoch_rows = []
with open(csv_path, newline="") as f:
    for row in csv.DictReader(f):
        batch = int(row["batch"])
        if batch > 0:
            steps.append(int(row["step"]))
            losses.append(float(row["train_loss"]))
        else:
            epoch_rows.append(row)
if not losses:
    raise SystemExit("no train_loss rows in metrics csv")

w, h = 960, 420
left, right, top, bottom = 64, 24, 32, 56
x_min, x_max = min(steps), max(steps)
y_min, y_max = min(losses), max(losses)
if math.isclose(y_min, y_max):
    y_min -= 1.0
    y_max += 1.0
pad = (y_max - y_min) * 0.08
y_min -= pad
y_max += pad

def sx(x):
    if x_max == x_min:
        return (left + w - right) / 2
    return left + (x - x_min) * (w - left - right) / (x_max - x_min)

def sy(y):
    return h - bottom - (y - y_min) * (h - top - bottom) / (y_max - y_min)

points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(steps, losses))
first, last = losses[0], losses[-1]
with open(svg_path, "w") as out:
    out.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n')
    out.write('<rect width="100%" height="100%" fill="white"/>\n')
    out.write(f'<line x1="{left}" y1="{h-bottom}" x2="{w-right}" y2="{h-bottom}" stroke="#222"/>\n')
    out.write(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{h-bottom}" stroke="#222"/>\n')
    out.write(f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2"/>\n')
    out.write(f'<text x="{left}" y="22" font-family="monospace" font-size="14">MNIST train loss: first={first:.4f}, last={last:.4f}, delta={last-first:.4f}</text>\n')
    out.write(f'<text x="{left}" y="{h-18}" font-family="monospace" font-size="12">step {x_min}..{x_max}</text>\n')
    out.write(f'<text x="10" y="{top+12}" font-family="monospace" font-size="12">loss {y_max:.3f}</text>\n')
    out.write(f'<text x="10" y="{h-bottom}" font-family="monospace" font-size="12">loss {y_min:.3f}</text>\n')
    out.write("</svg>\n")

print(f"mnist_loss_first={first:.6f}")
print(f"mnist_loss_last={last:.6f}")
print(f"mnist_loss_delta={last-first:.6f}")
for row in epoch_rows:
    train_eval_acc = row.get("train_eval_accuracy", "")
    train_eval_loss = row.get("train_eval_loss", "")
    train_eval_part = ""
    if train_eval_acc:
        train_eval_part = f" train_eval_loss={float(train_eval_loss):.6f} train_eval_acc={float(train_eval_acc):.6f}"
    print(
        "mnist_epoch_summary "
        f"epoch={row['epoch']} train_loss={float(row['train_loss']):.6f} "
        f"train_acc={float(row['train_accuracy']):.6f} "
        f"{train_eval_part} "
        f"test_loss={float(row['test_loss']):.6f} "
        f"test_acc={float(row['test_accuracy']):.6f}"
    )
print(f"metrics_csv={csv_path}")
print(f"loss_curve_svg={svg_path}")
PY
