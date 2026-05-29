#pragma once

#include <cstdint>
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
                                  bool average);

}  // namespace cverl::distributed
