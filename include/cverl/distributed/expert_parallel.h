#pragma once

#include <cstdint>
#include <vector>

#include <torch/torch.h>

#include "cverl/distributed/collectives.h"

namespace cverl::distributed {

struct ExpertParallelDispatchResult {
  torch::Tensor received_tokens;
  torch::Tensor send_counts;
  torch::Tensor recv_counts;
  int64_t capacity = 0;
};

torch::Tensor expert_parallel_all_to_all_autograd(const torch::Tensor& input,
                                                  Collectives& collectives,
                                                  const std::vector<int64_t>& group,
                                                  int64_t dim);

ExpertParallelDispatchResult expert_parallel_dispatch_equal_capacity(const torch::Tensor& tokens,
                                                                     const torch::Tensor& destination_ranks,
                                                                     Collectives& collectives,
                                                                     const std::vector<int64_t>& group);

}  // namespace cverl::distributed
