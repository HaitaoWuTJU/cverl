#include "cverl/distributed/weight_sync.h"

#include <stdexcept>

namespace cverl::distributed {

namespace {

void require_parameter(const ParameterView& p) {
  if (p.name.empty()) {
    throw std::invalid_argument("parameter name must not be empty");
  }
  if (!p.tensor.defined()) {
    throw std::invalid_argument("undefined parameter: " + p.name);
  }
  if (!p.tensor.is_contiguous()) {
    throw std::invalid_argument("parameter must be contiguous: " + p.name);
  }
}

}  // namespace

void broadcast_parameters_from_root(const std::vector<ParameterView>& parameters,
                                    Collectives& collectives,
                                    int64_t root,
                                    const std::vector<int64_t>& group) {
  if (parameters.empty()) {
    return;
  }
  torch::NoGradGuard no_grad;
  for (const auto& p : parameters) {
    require_parameter(p);
    auto synced = collectives.broadcast(p.tensor, root, group);
    if (synced.sizes() != p.tensor.sizes() || synced.scalar_type() != p.tensor.scalar_type()) {
      throw std::runtime_error("broadcast returned incompatible tensor for parameter: " + p.name);
    }
    p.tensor.copy_(synced);
  }
  collectives.barrier();
}

std::vector<ParameterView> module_parameter_views(torch::nn::Module& module, bool recurse) {
  std::vector<ParameterView> out;
  auto named = module.named_parameters(recurse);
  out.reserve(named.size());
  for (const auto& kv : named) {
    out.push_back(ParameterView{kv.key(), kv.value()});
  }
  return out;
}

}  // namespace cverl::distributed
