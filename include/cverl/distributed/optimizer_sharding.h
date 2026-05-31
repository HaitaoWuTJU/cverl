#pragma once

#include <cstdint>
#include <vector>

#include <torch/torch.h>

#include "cverl/distributed/collectives.h"
#include "cverl/torch/fp32_master_adamw.h"

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

struct FlatAdamWStepResult {
  FlatParameterShard gradient_shard;
  double local_grad_norm_sq = 0.0;
  double global_grad_norm = 0.0;
  double grad_clip_scale = 1.0;
  double local_grad_norm = 0.0;
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

FlatParameterShard reduce_scatter_flat_gradient_shard_bucketed(
    const std::vector<torch::Tensor>& parameters,
    Collectives& collectives,
    const std::vector<int64_t>& data_group,
    bool average,
    int64_t bucket_numel,
    bool require_grad = true);

torch::Tensor all_gather_flat_parameter_shards(const FlatParameterShard& local_shard,
                                               Collectives& collectives,
                                               const std::vector<int64_t>& data_group);

void all_gather_apply_flat_parameter_shard(const FlatParameterShard& local_shard,
                                           Collectives& collectives,
                                           const std::vector<int64_t>& data_group,
                                           const std::vector<torch::Tensor>& parameters);

void all_gather_apply_flat_parameter_shard_bucketed(const FlatParameterShard& local_shard,
                                                    Collectives& collectives,
                                                    const std::vector<int64_t>& data_group,
                                                    const std::vector<torch::Tensor>& parameters,
                                                    int64_t bucket_numel);

FlatAdamWStepResult flat_sharded_adamw_step(const std::vector<torch::Tensor>& parameters,
                                            FlatParameterShard& parameter_shard,
                                            cverl::torch_backend::FlatAdamW& optimizer,
                                            Collectives& data_collectives,
                                            const std::vector<int64_t>& data_group,
                                            Collectives& norm_collectives,
                                            const std::vector<int64_t>& norm_group,
                                            double max_grad_norm,
                                            bool average_gradients = true,
                                            bool require_grad = false,
                                            bool apply_parameters = true,
                                            int64_t reduce_scatter_bucket_numel = 0,
                                            int64_t all_gather_bucket_numel = 0);

void apply_flat_parameter_shard(const FlatParameterShard& shard,
                                const std::vector<torch::Tensor>& parameters);

void apply_full_flat_parameters(const torch::Tensor& flat_parameters,
                                int64_t original_numel,
                                const std::vector<torch::Tensor>& parameters);

}  // namespace cverl::distributed
