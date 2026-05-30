#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace cverl::rollout {

enum class GpuIpcDType : int32_t {
  Int64 = 0,
  Float32 = 1,
  UInt8 = 2,
  BFloat16 = 3,
  Float16 = 4,
};

enum class GpuRolloutTensorKind : int32_t {
  PromptIds = 0,
  ResponseIds = 1,
  ResponseMask = 2,
  OldLogProbs = 3,
  Rewards = 4,
  Advantages = 5,
  GroupIds = 6,
};

struct GpuIpcTensorHandle {
  GpuRolloutTensorKind kind = GpuRolloutTensorKind::PromptIds;
  GpuIpcDType dtype = GpuIpcDType::Int64;
  int32_t device_index = 0;
  int32_t ndim = 0;
  std::array<int64_t, 4> sizes{0, 0, 0, 0};
  std::array<int64_t, 4> strides{0, 0, 0, 0};
  uint64_t storage_offset_bytes = 0;
  uint64_t numel = 0;
  std::array<uint8_t, 64> cuda_ipc_handle{};
};

struct GpuIpcRolloutBatchManifest {
  int64_t step = 0;
  int64_t batch = 0;
  int64_t prompt_len = 0;
  int64_t response_len = 0;
  std::vector<GpuIpcTensorHandle> tensors;
};

GpuIpcTensorHandle export_cuda_ipc_tensor(const torch::Tensor& tensor,
                                          GpuRolloutTensorKind kind);

torch::Tensor import_cuda_ipc_tensor(const GpuIpcTensorHandle& handle,
                                     int32_t device_index = -1);

GpuIpcDType gpu_ipc_dtype_from_torch(torch::ScalarType dtype);
torch::ScalarType torch_dtype_from_gpu_ipc(GpuIpcDType dtype);

std::string cuda_ipc_handle_to_hex(const GpuIpcTensorHandle& handle);
void cuda_ipc_handle_from_hex(const std::string& hex, GpuIpcTensorHandle* handle);

}  // namespace cverl::rollout
