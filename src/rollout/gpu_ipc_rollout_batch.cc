#include "cverl/rollout/gpu_ipc_rollout_batch.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace cverl::rollout {
namespace {

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

int hex_value(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return 10 + c - 'a';
  if (c >= 'A' && c <= 'F') return 10 + c - 'A';
  throw std::invalid_argument("invalid hex character");
}

}  // namespace

GpuIpcDType gpu_ipc_dtype_from_torch(torch::ScalarType dtype) {
  if (dtype == torch::kInt64) return GpuIpcDType::Int64;
  if (dtype == torch::kFloat32) return GpuIpcDType::Float32;
  if (dtype == torch::kUInt8) return GpuIpcDType::UInt8;
  if (dtype == torch::kBFloat16) return GpuIpcDType::BFloat16;
  if (dtype == torch::kFloat16) return GpuIpcDType::Float16;
  throw std::invalid_argument("unsupported CUDA IPC rollout tensor dtype");
}

torch::ScalarType torch_dtype_from_gpu_ipc(GpuIpcDType dtype) {
  switch (dtype) {
    case GpuIpcDType::Int64:
      return torch::kInt64;
    case GpuIpcDType::Float32:
      return torch::kFloat32;
    case GpuIpcDType::UInt8:
      return torch::kUInt8;
    case GpuIpcDType::BFloat16:
      return torch::kBFloat16;
    case GpuIpcDType::Float16:
      return torch::kFloat16;
  }
  throw std::invalid_argument("unsupported CUDA IPC rollout tensor dtype");
}

GpuIpcTensorHandle export_cuda_ipc_tensor(const torch::Tensor& tensor,
                                          GpuRolloutTensorKind kind) {
  if (!tensor.defined() || !tensor.is_cuda() || !tensor.is_contiguous()) {
    throw std::invalid_argument("export_cuda_ipc_tensor requires a defined contiguous CUDA tensor");
  }
  if (tensor.dim() < 1 || tensor.dim() > 4) {
    throw std::invalid_argument("export_cuda_ipc_tensor supports tensor rank 1..4");
  }

  GpuIpcTensorHandle out;
  out.kind = kind;
  out.dtype = gpu_ipc_dtype_from_torch(tensor.scalar_type());
  out.device_index = tensor.get_device();
  out.ndim = static_cast<int32_t>(tensor.dim());
  out.numel = static_cast<uint64_t>(tensor.numel());
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    out.sizes[static_cast<size_t>(i)] = tensor.size(i);
    out.strides[static_cast<size_t>(i)] = tensor.stride(i);
  }

  cudaIpcMemHandle_t ipc_handle{};
  check_cuda(cudaSetDevice(out.device_index), "cudaSetDevice");
  check_cuda(cudaIpcGetMemHandle(&ipc_handle, tensor.data_ptr()), "cudaIpcGetMemHandle");
  static_assert(sizeof(ipc_handle) == 64, "cudaIpcMemHandle_t is expected to be 64 bytes");
  std::memcpy(out.cuda_ipc_handle.data(), &ipc_handle, sizeof(ipc_handle));
  return out;
}

torch::Tensor import_cuda_ipc_tensor(const GpuIpcTensorHandle& handle,
                                     int32_t device_index) {
  const int32_t target_device = device_index >= 0 ? device_index : handle.device_index;
  check_cuda(cudaSetDevice(target_device), "cudaSetDevice");

  cudaIpcMemHandle_t ipc_handle{};
  std::memcpy(&ipc_handle, handle.cuda_ipc_handle.data(), sizeof(ipc_handle));
  void* ptr = nullptr;
  check_cuda(cudaIpcOpenMemHandle(&ptr, ipc_handle, cudaIpcMemLazyEnablePeerAccess),
             "cudaIpcOpenMemHandle");

  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  sizes.reserve(static_cast<size_t>(handle.ndim));
  strides.reserve(static_cast<size_t>(handle.ndim));
  for (int32_t i = 0; i < handle.ndim; ++i) {
    sizes.push_back(handle.sizes[static_cast<size_t>(i)]);
    strides.push_back(handle.strides[static_cast<size_t>(i)]);
  }

  auto options = torch::TensorOptions()
                     .device(torch::Device(torch::kCUDA, target_device))
                     .dtype(torch_dtype_from_gpu_ipc(handle.dtype));
  return torch::from_blob(
      ptr,
      sizes,
      strides,
      [](void* p) {
        if (p != nullptr) {
          (void)cudaIpcCloseMemHandle(p);
        }
      },
      options);
}

std::string cuda_ipc_handle_to_hex(const GpuIpcTensorHandle& handle) {
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (uint8_t b : handle.cuda_ipc_handle) {
    out << std::setw(2) << static_cast<int>(b);
  }
  return out.str();
}

void cuda_ipc_handle_from_hex(const std::string& hex, GpuIpcTensorHandle* handle) {
  if (handle == nullptr) {
    throw std::invalid_argument("cuda_ipc_handle_from_hex requires output handle");
  }
  if (hex.size() != handle->cuda_ipc_handle.size() * 2) {
    throw std::invalid_argument("invalid CUDA IPC handle hex length");
  }
  for (size_t i = 0; i < handle->cuda_ipc_handle.size(); ++i) {
    int hi = hex_value(hex[2 * i]);
    int lo = hex_value(hex[2 * i + 1]);
    handle->cuda_ipc_handle[i] = static_cast<uint8_t>((hi << 4) | lo);
  }
}

}  // namespace cverl::rollout
