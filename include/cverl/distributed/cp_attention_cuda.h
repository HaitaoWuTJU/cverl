#pragma once

#include <cstdint>
#include <vector>

#include <torch/torch.h>

namespace cverl::distributed {

bool cp_attention_cuda_available();

torch::Tensor cp_ring_attention_cuda_forward(const torch::Tensor& query_local,
                                             const torch::Tensor& key_ring,
                                             const torch::Tensor& value_ring,
                                             const std::vector<int64_t>& key_begin_positions,
                                             int64_t query_begin,
                                             int64_t original_sequence_length,
                                             int64_t shard_size,
                                             double scale);

std::vector<torch::Tensor> cp_ring_attention_cuda_forward_with_lse(const torch::Tensor& query_local,
                                                                   const torch::Tensor& key_ring,
                                                                   const torch::Tensor& value_ring,
                                                                   const std::vector<int64_t>& key_begin_positions,
                                                                   int64_t query_begin,
                                                                   int64_t original_sequence_length,
                                                                   int64_t shard_size,
                                                                   double scale);

std::vector<torch::Tensor> cp_ring_attention_cuda_backward(const torch::Tensor& grad_out,
                                                           const torch::Tensor& query_local,
                                                           const torch::Tensor& key_ring,
                                                           const torch::Tensor& value_ring,
                                                           const std::vector<int64_t>& key_begin_positions,
                                                           int64_t query_begin,
                                                           int64_t original_sequence_length,
                                                           int64_t shard_size,
                                                           double scale);

std::vector<torch::Tensor> cp_ring_attention_cuda_backward_with_lse(const torch::Tensor& grad_out,
                                                                    const torch::Tensor& query_local,
                                                                    const torch::Tensor& key_ring,
                                                                    const torch::Tensor& value_ring,
                                                                    const torch::Tensor& query_lse,
                                                                    const std::vector<int64_t>& key_begin_positions,
                                                                    int64_t query_begin,
                                                                    int64_t original_sequence_length,
                                                                    int64_t shard_size,
                                                                    double scale);

}  // namespace cverl::distributed
