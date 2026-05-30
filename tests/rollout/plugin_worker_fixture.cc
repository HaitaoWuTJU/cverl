#include "cverl/rollout/worker.h"

#include <torch/torch.h>

namespace {

class FixtureWorker final : public cverl::rollout::RolloutWorker {
 public:
  FixtureWorker() {
    weight_ = torch::ones({2, 2}, torch::kFloat32).contiguous();
  }

  cverl::rollout::GenerationOutput generate(const cverl::rollout::TokenBatch& prompts,
                                            const cverl::rollout::GenerationConfig& config) override {
    cverl::rollout::GenerationOutput out;
    out.token_ids = torch::full({prompts.token_ids.size(0), config.max_new_tokens}, 7, torch::kLong);
    out.lengths = torch::full({prompts.token_ids.size(0)}, config.max_new_tokens, torch::kLong);
    return out;
  }

  std::vector<cverl::distributed::ParameterView> actor_parameters() override {
    return {cverl::distributed::ParameterView{"fixture_weight", weight_}};
  }

 private:
  torch::Tensor weight_;
};

}  // namespace

extern "C" cverl::rollout::RolloutWorker* cverl_create_rollout_worker(const char*) {
  return new FixtureWorker();
}

extern "C" void cverl_destroy_rollout_worker(cverl::rollout::RolloutWorker* worker) {
  delete worker;
}
