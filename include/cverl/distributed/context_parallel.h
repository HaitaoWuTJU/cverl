#pragma once

#include <torch/torch.h>

#include "cverl/distributed/collectives.h"
#include "cverl/distributed/topology.h"

namespace cverl::distributed {

struct ContextShardRange {
  int64_t begin = 0;
  int64_t end = 0;
  int64_t padded_end = 0;
};

struct ContextParallelRingStep {
  int64_t step = 0;
  int64_t kv_rank = 0;
  int64_t send_rank = -1;
  int64_t recv_rank = -1;
};

int64_t context_parallel_padded_length(int64_t sequence_length, int64_t context_parallel);

ContextShardRange context_parallel_sequence_range(int64_t sequence_length,
                                                  int64_t context_rank,
                                                  int64_t context_parallel);

torch::Tensor context_parallel_slice(const torch::Tensor& input,
                                     int64_t context_rank,
                                     int64_t context_parallel,
                                     int64_t sequence_dim);

torch::Tensor context_parallel_slice_padded(const torch::Tensor& input,
                                            int64_t context_rank,
                                            int64_t context_parallel,
                                            int64_t sequence_dim,
                                            double pad_value = 0.0);

torch::Tensor context_parallel_gather(const torch::Tensor& local,
                                      Collectives& collectives,
                                      const std::vector<int64_t>& context_group,
                                      int64_t sequence_dim);

torch::Tensor context_parallel_gather_autograd(const torch::Tensor& local,
                                               Collectives& collectives,
                                               const std::vector<int64_t>& context_group,
                                               int64_t sequence_dim);

torch::Tensor context_parallel_gather_padded(const torch::Tensor& local,
                                             Collectives& collectives,
                                             const std::vector<int64_t>& context_group,
                                             int64_t sequence_dim,
                                             int64_t original_sequence_length);

int64_t context_parallel_group_index(const std::vector<int64_t>& context_group, int64_t rank);

std::vector<ContextParallelRingStep> context_parallel_ring_schedule(const std::vector<int64_t>& context_group,
                                                                    int64_t rank);

torch::Tensor context_parallel_causal_attention(const torch::Tensor& query_local,
                                                const torch::Tensor& key_global,
                                                const torch::Tensor& value_global,
                                                int64_t query_begin,
                                                double scale);

torch::Tensor context_parallel_causal_attention_gather_kv(const torch::Tensor& query_local,
                                                          const torch::Tensor& key_local,
                                                          const torch::Tensor& value_local,
                                                          Collectives& collectives,
                                                          const std::vector<int64_t>& context_group,
                                                          int64_t context_rank,
                                                          int64_t original_sequence_length,
                                                          double scale);

}  // namespace cverl::distributed
