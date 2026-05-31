#pragma once

#include <cstdint>
#include <vector>

#include <torch/torch.h>

#include "cverl/distributed/collectives.h"

namespace cverl::distributed {

struct FlatParameterShardRange {
  int64_t parameter_index = 0;
  int64_t parameter_offset = 0;
  int64_t shard_offset = 0;
  int64_t numel = 0;
};

struct FlatParameterShard {
  torch::Tensor shard;
  int64_t original_numel = 0;
  int64_t padded_numel = 0;
  int64_t shard_begin = 0;
  int64_t shard_end = 0;
  std::vector<FlatParameterShardRange> ranges;
};

std::vector<int64_t> greedy_parameter_owner_by_size(const std::vector<int64_t>& parameter_bytes,
                                                    int64_t data_parallel);

std::vector<int64_t> owned_parameter_indices(const std::vector<int64_t>& owner_by_parameter,
                                             int64_t data_rank);

FlatParameterShard flatten_parameter_shard(const std::vector<torch::Tensor>& parameters,
                                           int64_t data_parallel,
                                           int64_t data_rank);

FlatParameterShard flatten_gradient_shard(const std::vector<torch::Tensor>& parameters,
                                          int64_t data_parallel,
                                          int64_t data_rank,
                                          bool require_grad = true);

FlatParameterShard reduce_scatter_flat_gradient_shard(const std::vector<torch::Tensor>& parameters,
                                                      Collectives& collectives,
                                                      const std::vector<int64_t>& data_group,
                                                      bool average,
                                                      bool require_grad = true);

void apply_flat_parameter_shard(const FlatParameterShard& shard,
                                const std::vector<torch::Tensor>& parameters);

void apply_full_flat_parameters(const torch::Tensor& flat_parameters,
                                int64_t original_numel,
                                const std::vector<torch::Tensor>& parameters);

}  // namespace cverl::distributed
