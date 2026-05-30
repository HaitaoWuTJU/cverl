#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

#include "cverl/distributed/collectives.h"

namespace cverl::distributed {

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

}  // namespace cverl::distributed
