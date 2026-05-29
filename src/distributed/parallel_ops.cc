#include "cverl/distributed/parallel_ops.h"

#include <stdexcept>

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
  return torch::matmul(input, weight.transpose(0, 1));
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
                                  bool average) {
  for (const auto& param : parameters) {
    if (!param.defined() || !param.requires_grad() || !param.grad().defined()) {
      continue;
    }
    auto grad = param.grad();
    auto synced = collectives.all_reduce(grad.contiguous(), average ? ReduceOp::Mean : ReduceOp::Sum, data_group);
    grad.copy_(synced);
  }
}

}  // namespace cverl::distributed
