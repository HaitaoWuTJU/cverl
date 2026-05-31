#include "cverl/distributed/parallel_ops.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace cverl::distributed {
namespace {

void validate_group(const ParallelGroup& group) {
  if (group.world_size <= 0 || group.rank < 0 || group.rank >= group.world_size) {
    throw std::invalid_argument("invalid parallel group rank/world_size");
  }
  if (static_cast<int64_t>(group.ranks.size()) != group.world_size) {
    throw std::invalid_argument("parallel group ranks must match world_size");
  }
}

torch::Tensor dense_linear(const torch::Tensor& input, const torch::Tensor& weight) {
  auto x = input.scalar_type() == weight.scalar_type() ? input : input.to(weight.scalar_type());
  return torch::matmul(x, weight.transpose(0, 1));
}

int64_t tensor_bytes(const torch::Tensor& tensor) {
  return tensor.numel() * static_cast<int64_t>(tensor.element_size());
}

struct GradBucketEntry {
  torch::Tensor grad;
  int64_t parameter_index = -1;
  int64_t numel = 0;
};

void flush_grad_bucket(std::vector<GradBucketEntry>* bucket,
                       Collectives& collectives,
                       const std::vector<int64_t>& data_group,
                       bool average) {
  if (bucket->empty()) {
    return;
  }

  torch::Tensor flat;
  if (bucket->size() == 1) {
    const auto& entry = bucket->front();
    flat = entry.grad.contiguous().view({entry.numel});
  } else {
    std::vector<torch::Tensor> flat_grads;
    flat_grads.reserve(bucket->size());
    for (const auto& entry : *bucket) {
      flat_grads.push_back(entry.grad.contiguous().view({entry.numel}));
    }
    flat = torch::cat(flat_grads, 0).contiguous();
  }
  auto synced = collectives.all_reduce(flat, average ? ReduceOp::Mean : ReduceOp::Sum, data_group).contiguous();
  int64_t offset = 0;
  for (const auto& entry : *bucket) {
    auto shard = synced.narrow(0, offset, entry.numel).view(entry.grad.sizes());
    entry.grad.copy_(shard);
    offset += entry.numel;
  }
  bucket->clear();
}

void flush_grad_reduce_scatter_bucket(std::vector<GradBucketEntry>* bucket,
                                      Collectives& collectives,
                                      const std::vector<int64_t>& data_group,
                                      bool average,
                                      std::vector<GradientReduceScatterBucket>* out) {
  if (bucket->empty()) {
    return;
  }
  if (data_group.empty()) {
    throw std::invalid_argument("data_parallel_reduce_scatter_gradients requires non-empty data group");
  }

  int64_t original_numel = 0;
  GradientReduceScatterBucket meta;
  meta.parameter_indices.reserve(bucket->size());
  meta.parameter_numels.reserve(bucket->size());
  for (const auto& entry : *bucket) {
    original_numel += entry.numel;
    meta.parameter_indices.push_back(entry.parameter_index);
    meta.parameter_numels.push_back(entry.numel);
  }
  torch::Tensor flat;
  if (bucket->size() == 1) {
    const auto& entry = bucket->front();
    flat = entry.grad.contiguous().view({entry.numel});
  } else {
    std::vector<torch::Tensor> flat_grads;
    flat_grads.reserve(bucket->size());
    for (const auto& entry : *bucket) {
      flat_grads.push_back(entry.grad.contiguous().view({entry.numel}));
    }
    flat = torch::cat(flat_grads, 0).contiguous();
  }
  const int64_t group_size = static_cast<int64_t>(data_group.size());
  const int64_t remainder = flat.numel() % group_size;
  const int64_t pad = remainder == 0 ? 0 : group_size - remainder;
  if (pad > 0) {
    flat = torch::cat({flat, torch::zeros({pad}, flat.options())}, 0).contiguous();
  }
  auto shard = collectives.reduce_scatter(
      flat, average ? ReduceOp::Mean : ReduceOp::Sum, data_group, 0).contiguous();
  meta.shard = shard;
  meta.original_numel = original_numel;
  meta.padded_numel = flat.numel();
  out->push_back(std::move(meta));
  bucket->clear();
}

}  // namespace

torch::Tensor shard_dim(const torch::Tensor& tensor, int64_t dim, int64_t rank, int64_t world_size) {
  if (world_size <= 0 || rank < 0 || rank >= world_size) {
    throw std::invalid_argument("invalid shard rank/world_size");
  }
  if (dim < 0) {
    dim += tensor.dim();
  }
  if (dim < 0 || dim >= tensor.dim()) {
    throw std::invalid_argument("shard dim out of range");
  }
  int64_t size = tensor.size(dim);
  if (size % world_size != 0) {
    throw std::invalid_argument("tensor dimension is not divisible by world_size");
  }
  int64_t shard = size / world_size;
  return tensor.narrow(dim, rank * shard, shard).contiguous();
}

torch::Tensor column_parallel_linear(const torch::Tensor& input,
                                     const torch::Tensor& full_weight,
                                     const ParallelGroup& group) {
  validate_group(group);
  auto weight_shard = shard_dim(full_weight, 0, group.rank, group.world_size);
  return dense_linear(input, weight_shard);
}

torch::Tensor row_parallel_linear(const torch::Tensor& input,
                                  const torch::Tensor& full_weight,
                                  const ParallelGroup& group,
                                  bool reduce) {
  validate_group(group);
  auto input_shard = shard_dim(input, -1, group.rank, group.world_size);
  auto weight_shard = shard_dim(full_weight, 1, group.rank, group.world_size);
  auto partial = dense_linear(input_shard, weight_shard);
  if (!reduce || group.world_size == 1) {
    return partial;
  }
  if (group.collectives == nullptr) {
    throw std::invalid_argument("row_parallel_linear reduction requires collectives");
  }
  return group.collectives->all_reduce(partial.contiguous(), ReduceOp::Sum, group.ranks);
}

torch::Tensor tensor_parallel_mlp_swiglu(const torch::Tensor& input,
                                         const torch::Tensor& gate_weight,
                                         const torch::Tensor& up_weight,
                                         const torch::Tensor& down_weight,
                                         const ParallelGroup& group) {
  validate_group(group);
  auto gate = column_parallel_linear(input, gate_weight, group);
  auto up = column_parallel_linear(input, up_weight, group);
  auto hidden_shard = torch::silu(gate) * up;
  auto down_weight_shard = shard_dim(down_weight, 1, group.rank, group.world_size);
  auto partial = dense_linear(hidden_shard, down_weight_shard);
  if (group.world_size == 1) {
    return partial;
  }
  if (group.collectives == nullptr) {
    throw std::invalid_argument("tensor_parallel_mlp_swiglu requires collectives");
  }
  return group.collectives->all_reduce(partial.contiguous(), ReduceOp::Sum, group.ranks);
}

void data_parallel_sync_gradients(const std::vector<torch::Tensor>& parameters,
                                  Collectives& collectives,
                                  const std::vector<int64_t>& data_group,
                                  bool average,
                                  int64_t bucket_bytes) {
  if (bucket_bytes <= 0) {
    throw std::invalid_argument("data_parallel_sync_gradients bucket_bytes must be positive");
  }
  std::vector<GradBucketEntry> bucket;
  bucket.reserve(parameters.size());
  int64_t current_bytes = 0;
  torch::ScalarType current_dtype = torch::kFloat32;
  torch::Device current_device(torch::kCPU);
  bool have_bucket = false;

  for (const auto& param : parameters) {
    if (!param.defined() || !param.requires_grad() || !param.grad().defined()) {
      continue;
    }
    auto grad = param.grad();
    const int64_t grad_bytes = tensor_bytes(grad);
    const bool incompatible =
        have_bucket && (grad.scalar_type() != current_dtype || grad.device() != current_device);
    const bool would_overflow = have_bucket && current_bytes + grad_bytes > bucket_bytes;
    if (incompatible || would_overflow) {
      flush_grad_bucket(&bucket, collectives, data_group, average);
      current_bytes = 0;
      have_bucket = false;
    }

    if (!have_bucket) {
      current_dtype = grad.scalar_type();
      current_device = grad.device();
      have_bucket = true;
    }
    bucket.push_back(GradBucketEntry{grad, -1, grad.numel()});
    current_bytes += grad_bytes;
  }
  flush_grad_bucket(&bucket, collectives, data_group, average);
}

std::vector<GradientReduceScatterBucket> data_parallel_reduce_scatter_gradients(
    const std::vector<torch::Tensor>& parameters,
    Collectives& collectives,
    const std::vector<int64_t>& data_group,
    bool average,
    int64_t bucket_bytes) {
  if (bucket_bytes <= 0) {
    throw std::invalid_argument("data_parallel_reduce_scatter_gradients bucket_bytes must be positive");
  }
  std::vector<GradientReduceScatterBucket> out;
  std::vector<GradBucketEntry> bucket;
  bucket.reserve(parameters.size());
  int64_t current_bytes = 0;
  torch::ScalarType current_dtype = torch::kFloat32;
  torch::Device current_device(torch::kCPU);
  bool have_bucket = false;

  for (size_t i = 0; i < parameters.size(); ++i) {
    const auto& param = parameters[i];
    if (!param.defined() || !param.requires_grad() || !param.grad().defined()) {
      continue;
    }
    auto grad = param.grad();
    const int64_t grad_bytes = tensor_bytes(grad);
    const bool incompatible =
        have_bucket && (grad.scalar_type() != current_dtype || grad.device() != current_device);
    const bool would_overflow = have_bucket && current_bytes + grad_bytes > bucket_bytes;
    if (incompatible || would_overflow) {
      flush_grad_reduce_scatter_bucket(&bucket, collectives, data_group, average, &out);
      current_bytes = 0;
      have_bucket = false;
    }

    if (!have_bucket) {
      current_dtype = grad.scalar_type();
      current_device = grad.device();
      have_bucket = true;
    }
    bucket.push_back(GradBucketEntry{grad, static_cast<int64_t>(i), grad.numel()});
    current_bytes += grad_bytes;
  }
  flush_grad_reduce_scatter_bucket(&bucket, collectives, data_group, average, &out);
  return out;
}

}  // namespace cverl::distributed
