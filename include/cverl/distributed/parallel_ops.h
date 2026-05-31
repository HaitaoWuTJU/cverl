#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <torch/torch.h>

#include "cverl/distributed/collectives.h"

namespace cverl::distributed {

struct ParallelGroup {
  int64_t rank = 0;
  int64_t world_size = 1;
  std::vector<int64_t> ranks{0};
  Collectives* collectives = nullptr;
};

struct GradientReduceScatterBucket {
  torch::Tensor shard;
  int64_t original_numel = 0;
  int64_t padded_numel = 0;
  std::vector<int64_t> parameter_indices;
  std::vector<int64_t> parameter_numels;
};

torch::Tensor shard_dim(const torch::Tensor& tensor, int64_t dim, int64_t rank, int64_t world_size);

torch::Tensor column_parallel_linear(const torch::Tensor& input,
                                     const torch::Tensor& full_weight,
                                     const ParallelGroup& group);

torch::Tensor row_parallel_linear(const torch::Tensor& input,
                                  const torch::Tensor& full_weight,
                                  const ParallelGroup& group,
                                  bool reduce = true);

torch::Tensor tensor_parallel_mlp_swiglu(const torch::Tensor& input,
                                         const torch::Tensor& gate_weight,
                                         const torch::Tensor& up_weight,
                                         const torch::Tensor& down_weight,
                                         const ParallelGroup& group);

void data_parallel_sync_gradients(const std::vector<torch::Tensor>& parameters,
                                  Collectives& collectives,
                                  const std::vector<int64_t>& data_group,
                                  bool average,
                                  int64_t bucket_bytes = 25 * 1024 * 1024,
                                  std::optional<torch::ScalarType> communication_dtype = std::nullopt);

std::vector<GradientReduceScatterBucket> data_parallel_reduce_scatter_gradients(
    const std::vector<torch::Tensor>& parameters,
    Collectives& collectives,
    const std::vector<int64_t>& data_group,
    bool average,
    int64_t bucket_bytes = 25 * 1024 * 1024,
    std::optional<torch::ScalarType> communication_dtype = std::nullopt);

}  // namespace cverl::distributed
