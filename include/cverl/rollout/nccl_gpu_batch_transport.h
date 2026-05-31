#pragma once

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "cverl/distributed/nccl_collectives.h"
#include "cverl/rollout/gpu_ipc_rollout_batch.h"

namespace cverl::rollout {

struct GpuRolloutTensorDescriptor {
  GpuRolloutTensorKind kind = GpuRolloutTensorKind::PromptIds;
  GpuIpcDType dtype = GpuIpcDType::Int64;
  std::vector<int64_t> shape;
};

struct GpuRolloutBatchDescriptor {
  int64_t step = 0;
  int64_t batch = 0;
  int64_t prompt_len = 0;
  int64_t response_len = 0;
  int64_t weight_version = 0;
  std::vector<GpuRolloutTensorDescriptor> tensors;
};

struct GpuRolloutBatch {
  GpuRolloutBatchDescriptor descriptor;
  torch::Tensor prompt_ids;
  torch::Tensor response_ids;
  torch::Tensor response_mask;
  torch::Tensor advantages;
  torch::Tensor old_log_probs;
  torch::Tensor ref_log_probs;
  torch::Tensor rewards;
  torch::Tensor group_ids;
};

GpuRolloutBatchDescriptor describe_gpu_rollout_batch(const GpuRolloutBatch& batch);

class NCCLGpuBatchSender {
 public:
  explicit NCCLGpuBatchSender(distributed::NcclCollectives& comm) : comm_(comm) {}
  void send(const GpuRolloutBatch& batch, int64_t peer);

 private:
  distributed::NcclCollectives& comm_;
};

class NCCLGpuBatchReceiver {
 public:
  NCCLGpuBatchReceiver(distributed::NcclCollectives& comm, int device_index)
      : comm_(comm), device_index_(device_index) {}
  GpuRolloutBatch recv(int64_t peer);

 private:
  distributed::NcclCollectives& comm_;
  int device_index_;
};

class TrainerIngressQueue {
 public:
  explicit TrainerIngressQueue(size_t capacity) : capacity_(capacity) {}
  bool push(GpuRolloutBatch batch);
  std::optional<GpuRolloutBatch> pop();
  size_t size() const { return queue_.size(); }
  size_t capacity() const { return capacity_; }

 private:
  size_t capacity_;
  std::deque<GpuRolloutBatch> queue_;
};

}  // namespace cverl::rollout
