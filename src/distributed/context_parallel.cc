#include "cverl/distributed/context_parallel.h"

#include <stdexcept>

namespace cverl::distributed {

torch::Tensor context_parallel_slice(const torch::Tensor& input,
                                     int64_t context_rank,
                                     int64_t context_parallel,
                                     int64_t sequence_dim) {
  if (!input.defined()) {
    throw std::invalid_argument("context_parallel_slice input must be defined");
  }
  if (context_parallel <= 0 || context_rank < 0 || context_rank >= context_parallel) {
    throw std::invalid_argument("invalid context parallel rank/size");
  }
  int64_t dim = sequence_dim;
  if (dim < 0) {
    dim += input.dim();
  }
  if (dim < 0 || dim >= input.dim()) {
    throw std::invalid_argument("sequence_dim out of range");
  }
  const int64_t seq = input.size(dim);
  if (seq % context_parallel != 0) {
    throw std::invalid_argument("sequence length must be divisible by context_parallel");
  }
  const int64_t shard = seq / context_parallel;
  return input.narrow(dim, context_rank * shard, shard).contiguous();
}

torch::Tensor context_parallel_gather(const torch::Tensor& local,
                                      Collectives& collectives,
                                      const std::vector<int64_t>& context_group,
                                      int64_t sequence_dim) {
  if (context_group.size() <= 1) {
    return local;
  }
  int64_t dim = sequence_dim;
  if (dim < 0) {
    dim += local.dim();
  }
  if (dim < 0 || dim >= local.dim()) {
    throw std::invalid_argument("sequence_dim out of range");
  }
  if (dim == 0) {
    return collectives.all_gather(local.contiguous(), context_group, 0);
  }
  auto moved = local.transpose(0, dim).contiguous();
  auto gathered = collectives.all_gather(moved, context_group, 0);
  return gathered.transpose(0, dim).contiguous();
}

}  // namespace cverl::distributed
