#include "cverl/rollout/worker.h"

namespace cverl::rollout {

void synchronize_rollout_actor_weights(
    RolloutWorker& worker,
    cverl::distributed::Collectives& collectives,
    int64_t trainer_root,
    const std::vector<int64_t>& group) {
  cverl::distributed::broadcast_parameters_from_root(
      worker.actor_parameters(), collectives, trainer_root, group);
}

}  // namespace cverl::rollout
