#include "cverl/distributed/weight_sync.h"

#ifdef CVERL_ENABLE_NCCL
#include "cverl/distributed/nccl_collectives.h"
#endif

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

#ifdef CVERL_ENABLE_NCCL
void send_parameter_shards(const std::vector<ParameterView>& shards,
                           NcclCollectives& collectives,
                           int64_t peer) {
  torch::NoGradGuard no_grad;
  for (const auto& shard : shards) {
    require_parameter(shard);
    collectives.send(shard.tensor.contiguous(), peer);
  }
}

void recv_parameter_shards(const std::vector<ParameterView>& shards,
                           NcclCollectives& collectives,
                           int64_t peer) {
  torch::NoGradGuard no_grad;
  for (const auto& shard : shards) {
    require_parameter(shard);
    auto received = collectives.recv_like(shard.tensor.contiguous(), peer);
    if (received.sizes() != shard.tensor.sizes() || received.scalar_type() != shard.tensor.scalar_type()) {
      throw std::runtime_error("received incompatible parameter shard: " + shard.name);
    }
    shard.tensor.copy_(received);
  }
}
#endif

}  // namespace cverl::distributed
