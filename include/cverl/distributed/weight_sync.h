#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

#include "cverl/distributed/collectives.h"

namespace cverl::distributed {

#ifdef CVERL_ENABLE_NCCL
class NcclCollectives;
#endif

struct ParameterView {
  std::string name;
  torch::Tensor tensor;
};

// Broadcast parameters in-place from `root` to every rank in `group`.
// All ranks must pass the same parameter list order, names, shapes, dtypes,
// and devices. Tensor storage is never serialized through CPU memory.
void broadcast_parameters_from_root(const std::vector<ParameterView>& parameters,
                                    Collectives& collectives,
                                    int64_t root,
                                    const std::vector<int64_t>& group);

std::vector<ParameterView> module_parameter_views(torch::nn::Module& module, bool recurse = true);

#ifdef CVERL_ENABLE_NCCL
// Shard-wise GPU parameter transfer. The caller is responsible for building
// matching source/destination shard lists from a reshard plan. No parameter is
// gathered to CPU or reconstructed as a full model tensor here.
void send_parameter_shards(const std::vector<ParameterView>& shards,
                           NcclCollectives& collectives,
                           int64_t peer,
                           int64_t bucket_bytes = 25 * 1024 * 1024);

void recv_parameter_shards(const std::vector<ParameterView>& shards,
                           NcclCollectives& collectives,
                           int64_t peer,
                           int64_t bucket_bytes = 25 * 1024 * 1024);
#endif

}  // namespace cverl::distributed
