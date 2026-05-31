#include "cverl/rollout/nccl_gpu_batch_transport.h"

#include <stdexcept>
#include <unordered_map>

namespace cverl::rollout {
namespace {

constexpr int64_t kMagic = 0x4356524c42415443LL;  // CVRLBATC
constexpr int64_t kVersion = 1;
constexpr int64_t kMaxTensors = 8;
constexpr int64_t kWordsPerTensor = 8;  // kind,dtype,ndim,shape[0..3],reserved
constexpr int64_t kHeaderWords = 8 + kMaxTensors * kWordsPerTensor;

bool tensor_defined(const torch::Tensor& t) {
  return t.defined() && t.numel() > 0;
}

GpuIpcDType dtype_of(const torch::Tensor& tensor) {
  return gpu_ipc_dtype_from_torch(tensor.scalar_type());
}

void require_cuda_payload(const torch::Tensor& tensor, const char* name) {
  if (!tensor.defined() || !tensor.is_cuda() || !tensor.is_contiguous()) {
    throw std::invalid_argument(std::string(name) + " must be a contiguous CUDA tensor");
  }
}

void add_desc(std::vector<GpuRolloutTensorDescriptor>* out,
              GpuRolloutTensorKind kind,
              const torch::Tensor& tensor) {
  if (!tensor_defined(tensor)) {
    return;
  }
  require_cuda_payload(tensor, "GpuRolloutBatch tensor");
  out->push_back(GpuRolloutTensorDescriptor{
      kind,
      dtype_of(tensor),
      tensor.sizes().vec(),
  });
}

torch::Tensor tensor_for_kind(const GpuRolloutBatch& batch, GpuRolloutTensorKind kind) {
  switch (kind) {
    case GpuRolloutTensorKind::PromptIds:
      return batch.prompt_ids;
    case GpuRolloutTensorKind::ResponseIds:
      return batch.response_ids;
    case GpuRolloutTensorKind::ResponseMask:
      return batch.response_mask;
    case GpuRolloutTensorKind::OldLogProbs:
      return batch.old_log_probs;
    case GpuRolloutTensorKind::RefLogProbs:
      return batch.ref_log_probs;
    case GpuRolloutTensorKind::Rewards:
      return batch.rewards;
    case GpuRolloutTensorKind::Advantages:
      return batch.advantages;
    case GpuRolloutTensorKind::GroupIds:
      return batch.group_ids;
  }
  throw std::invalid_argument("unknown rollout tensor kind");
}

void set_tensor_for_kind(GpuRolloutBatch* batch, GpuRolloutTensorKind kind, torch::Tensor tensor) {
  switch (kind) {
    case GpuRolloutTensorKind::PromptIds:
      batch->prompt_ids = std::move(tensor);
      return;
    case GpuRolloutTensorKind::ResponseIds:
      batch->response_ids = std::move(tensor);
      return;
    case GpuRolloutTensorKind::ResponseMask:
      batch->response_mask = std::move(tensor);
      return;
    case GpuRolloutTensorKind::OldLogProbs:
      batch->old_log_probs = std::move(tensor);
      return;
    case GpuRolloutTensorKind::RefLogProbs:
      batch->ref_log_probs = std::move(tensor);
      return;
    case GpuRolloutTensorKind::Rewards:
      batch->rewards = std::move(tensor);
      return;
    case GpuRolloutTensorKind::Advantages:
      batch->advantages = std::move(tensor);
      return;
    case GpuRolloutTensorKind::GroupIds:
      batch->group_ids = std::move(tensor);
      return;
  }
  throw std::invalid_argument("unknown rollout tensor kind");
}

torch::Tensor pack_header(const GpuRolloutBatchDescriptor& desc, int device_index) {
  if (static_cast<int64_t>(desc.tensors.size()) > kMaxTensors) {
    throw std::invalid_argument("too many tensors in GpuRolloutBatchDescriptor");
  }
  auto cpu = torch::zeros({kHeaderWords}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  auto* p = cpu.data_ptr<int64_t>();
  p[0] = kMagic;
  p[1] = kVersion;
  p[2] = desc.step;
  p[3] = desc.batch;
  p[4] = desc.prompt_len;
  p[5] = desc.response_len;
  p[6] = desc.weight_version;
  p[7] = static_cast<int64_t>(desc.tensors.size());
  for (size_t i = 0; i < desc.tensors.size(); ++i) {
    const auto& t = desc.tensors[i];
    const size_t off = 8 + i * kWordsPerTensor;
    p[off + 0] = static_cast<int64_t>(t.kind);
    p[off + 1] = static_cast<int64_t>(t.dtype);
    p[off + 2] = static_cast<int64_t>(t.shape.size());
    if (t.shape.size() > 4) {
      throw std::invalid_argument("rollout NCCL tensor rank > 4");
    }
    for (size_t d = 0; d < t.shape.size(); ++d) {
      p[off + 3 + d] = t.shape[d];
    }
  }
  return cpu.to(torch::Device(torch::kCUDA, device_index), /*non_blocking=*/false).contiguous();
}

GpuRolloutBatchDescriptor unpack_header(const torch::Tensor& header) {
  auto cpu = header.to(torch::kCPU).contiguous();
  const auto* p = cpu.data_ptr<int64_t>();
  if (p[0] != kMagic || p[1] != kVersion) {
    throw std::runtime_error("invalid NCCL GPU rollout batch header");
  }
  GpuRolloutBatchDescriptor desc;
  desc.step = p[2];
  desc.batch = p[3];
  desc.prompt_len = p[4];
  desc.response_len = p[5];
  desc.weight_version = p[6];
  const int64_t n = p[7];
  if (n < 0 || n > kMaxTensors) {
    throw std::runtime_error("invalid tensor count in NCCL GPU rollout batch header");
  }
  desc.tensors.reserve(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    const size_t off = 8 + static_cast<size_t>(i) * kWordsPerTensor;
    GpuRolloutTensorDescriptor t;
    t.kind = static_cast<GpuRolloutTensorKind>(p[off + 0]);
    t.dtype = static_cast<GpuIpcDType>(p[off + 1]);
    const int64_t ndim = p[off + 2];
    if (ndim <= 0 || ndim > 4) {
      throw std::runtime_error("invalid tensor rank in NCCL GPU rollout batch header");
    }
    t.shape.reserve(static_cast<size_t>(ndim));
    for (int64_t d = 0; d < ndim; ++d) {
      t.shape.push_back(p[off + 3 + d]);
    }
    desc.tensors.push_back(std::move(t));
  }
  return desc;
}

torch::Tensor empty_from_desc(const GpuRolloutTensorDescriptor& desc, int device_index) {
  return torch::empty(desc.shape,
                      torch::TensorOptions()
                          .device(torch::Device(torch::kCUDA, device_index))
                          .dtype(torch_dtype_from_gpu_ipc(desc.dtype)))
      .contiguous();
}

}  // namespace

GpuRolloutBatchDescriptor describe_gpu_rollout_batch(const GpuRolloutBatch& batch) {
  GpuRolloutBatchDescriptor desc = batch.descriptor;
  if (desc.batch <= 0 && tensor_defined(batch.prompt_ids)) {
    desc.batch = batch.prompt_ids.size(0);
  }
  if (desc.prompt_len <= 0 && tensor_defined(batch.prompt_ids)) {
    desc.prompt_len = batch.prompt_ids.size(1);
  }
  if (desc.response_len <= 0 && tensor_defined(batch.response_ids)) {
    desc.response_len = batch.response_ids.size(1);
  }
  desc.tensors.clear();
  add_desc(&desc.tensors, GpuRolloutTensorKind::PromptIds, batch.prompt_ids);
  add_desc(&desc.tensors, GpuRolloutTensorKind::ResponseIds, batch.response_ids);
  add_desc(&desc.tensors, GpuRolloutTensorKind::ResponseMask, batch.response_mask);
  add_desc(&desc.tensors, GpuRolloutTensorKind::OldLogProbs, batch.old_log_probs);
  add_desc(&desc.tensors, GpuRolloutTensorKind::RefLogProbs, batch.ref_log_probs);
  add_desc(&desc.tensors, GpuRolloutTensorKind::Rewards, batch.rewards);
  add_desc(&desc.tensors, GpuRolloutTensorKind::Advantages, batch.advantages);
  add_desc(&desc.tensors, GpuRolloutTensorKind::GroupIds, batch.group_ids);
  return desc;
}

void NCCLGpuBatchSender::send(const GpuRolloutBatch& batch, int64_t peer) {
  auto desc = describe_gpu_rollout_batch(batch);
  int device = batch.prompt_ids.defined() ? batch.prompt_ids.get_device() : 0;
  auto header = pack_header(desc, device);
  comm_.send(header.contiguous(), peer);
  for (const auto& t : desc.tensors) {
    comm_.send(tensor_for_kind(batch, t.kind).contiguous(), peer);
  }
}

GpuRolloutBatch NCCLGpuBatchReceiver::recv(int64_t peer) {
  auto header_like = torch::empty({kHeaderWords},
                                  torch::TensorOptions()
                                      .device(torch::Device(torch::kCUDA, device_index_))
                                      .dtype(torch::kInt64));
  auto header = comm_.recv_like(header_like.contiguous(), peer);
  GpuRolloutBatch batch;
  batch.descriptor = unpack_header(header);
  for (const auto& t : batch.descriptor.tensors) {
    auto like = empty_from_desc(t, device_index_);
    auto received = comm_.recv_like(like, peer);
    set_tensor_for_kind(&batch, t.kind, received);
  }
  return batch;
}

bool TrainerIngressQueue::push(GpuRolloutBatch batch) {
  if (queue_.size() >= capacity_) {
    return false;
  }
  queue_.push_back(std::move(batch));
  return true;
}

std::optional<GpuRolloutBatch> TrainerIngressQueue::pop() {
  if (queue_.empty()) {
    return std::nullopt;
  }
  auto out = std::move(queue_.front());
  queue_.pop_front();
  return out;
}

}  // namespace cverl::rollout
