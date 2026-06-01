#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/data/workspace/ceph/zw/rl/datasets/cifar10}"
PARQUET_DIR="${PARQUET_DIR:-${DATA_ROOT}/hf_parquet}"
TENSOR_DIR="${TENSOR_DIR:-${DATA_ROOT}/hf_tensors}"
HF_CIFAR10_TRAIN_URL="${HF_CIFAR10_TRAIN_URL:-https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet?download=true}"
HF_CIFAR10_TEST_URL="${HF_CIFAR10_TEST_URL:-https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet?download=true}"
TRAIN_PARQUET="${TRAIN_PARQUET:-${PARQUET_DIR}/train-00000-of-00001.parquet}"
TEST_PARQUET="${TEST_PARQUET:-${PARQUET_DIR}/test-00000-of-00001.parquet}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"

mkdir -p "${PARQUET_DIR}" "${TENSOR_DIR}"

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  "${PYTHON}" -m pip install pyarrow
fi

if [[ ! -s "${TRAIN_PARQUET}" ]]; then
  tmp="${TRAIN_PARQUET}.tmp"
  rm -f "${tmp}"
  curl -fL --connect-timeout 20 --max-time "${CURL_MAX_TIME:-900}" \
    --retry 5 --retry-delay 2 \
    -o "${tmp}" "${HF_CIFAR10_TRAIN_URL}"
  mv "${tmp}" "${TRAIN_PARQUET}"
fi

if [[ ! -s "${TEST_PARQUET}" ]]; then
  tmp="${TEST_PARQUET}.tmp"
  rm -f "${tmp}"
  curl -fL --connect-timeout 20 --max-time "${CURL_MAX_TIME:-900}" \
    --retry 5 --retry-delay 2 \
    -o "${tmp}" "${HF_CIFAR10_TEST_URL}"
  mv "${tmp}" "${TEST_PARQUET}"
fi

"${PYTHON}" tools/bench/export_hf_cifar10_tensors.py \
  --parquet "${TRAIN_PARQUET}" \
  --out-dir "${TENSOR_DIR}" \
  --split train

"${PYTHON}" tools/bench/export_hf_cifar10_tensors.py \
  --parquet "${TEST_PARQUET}" \
  --out-dir "${TENSOR_DIR}" \
  --split test

echo "cifar10_tensor_dir=${TENSOR_DIR}"
