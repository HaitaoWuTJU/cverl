#pragma once

#include <torch/torch.h>

#include "cverl/distributed/collectives.h"
#include "cverl/distributed/topology.h"

namespace cverl::distributed {

torch::Tensor context_parallel_slice(const torch::Tensor& input,
                                     int64_t context_rank,
                                     int64_t context_parallel,
                                     int64_t sequence_dim);

torch::Tensor context_parallel_gather(const torch::Tensor& local,
                                      Collectives& collectives,
                                      const std::vector<int64_t>& context_group,
                                      int64_t sequence_dim);

}  // namespace cverl::distributed
