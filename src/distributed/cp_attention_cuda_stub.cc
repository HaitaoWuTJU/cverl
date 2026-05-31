#include "cverl/distributed/cp_attention_cuda.h"

#include <stdexcept>

namespace cverl::distributed {

#ifndef CVERL_ENABLE_CUDA
bool cp_attention_cuda_available() {
  return false;
}

torch::Tensor cp_ring_attention_cuda_forward(const torch::Tensor& query_local,
                                             const torch::Tensor& key_ring,
                                             const torch::Tensor& value_ring,
                                             const std::vector<int64_t>& key_begin_positions,
                                             int64_t query_begin,
                                             int64_t original_sequence_length,
                                             int64_t shard_size,
                                             double scale) {
  (void)query_local;
  (void)key_ring;
  (void)value_ring;
  (void)key_begin_positions;
  (void)query_begin;
  (void)original_sequence_length;
  (void)shard_size;
  (void)scale;
  throw std::runtime_error("CP attention CUDA kernel was not built");
}
#endif

}  // namespace cverl::distributed
