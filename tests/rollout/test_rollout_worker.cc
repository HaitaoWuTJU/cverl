#include "cverl/rollout/worker.h"

#include <torch/torch.h>

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool cond, const std::string& msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

class FakeWorker final : public cverl::rollout::RolloutWorker {
 public:
  FakeWorker() {
    actor_weight_ = torch::arange(6, torch::kFloat32).reshape({2, 3}).contiguous();
  }

  cverl::rollout::GenerationOutput generate(const cverl::rollout::TokenBatch& prompts,
                                            const cverl::rollout::GenerationConfig& config) override {
    cverl::rollout::GenerationOutput out;
    out.token_ids = torch::full({prompts.token_ids.size(0), config.max_new_tokens}, 1, torch::kInt64);
    out.lengths = torch::full({prompts.token_ids.size(0)}, config.max_new_tokens, torch::kInt64);
    return out;
  }

  std::vector<cverl::distributed::ParameterView> actor_parameters() override {
    return {cverl::distributed::ParameterView{"actor_weight", actor_weight_}};
  }

  torch::Tensor actor_weight_;
};

}  // namespace

int main() {
  try {
    FakeWorker worker;
    cverl::distributed::SingleProcessCollectives collectives;
    cverl::rollout::synchronize_rollout_actor_weights(worker, collectives, 0, {0});
    require(torch::allclose(worker.actor_weight_, torch::arange(6, torch::kFloat32).reshape({2, 3})),
            "actor weight unchanged after single-rank sync");

    cverl::rollout::TokenBatch prompts;
    prompts.token_ids = torch::ones({2, 4}, torch::kInt64);
    cverl::rollout::GenerationConfig config;
    config.max_new_tokens = 3;
    auto out = worker.generate(prompts, config);
    require(out.token_ids.dim() == 2 && out.token_ids.size(0) == 2 && out.token_ids.size(1) == 3,
            "generated token shape");

    std::cout << "rollout worker tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_rollout_worker failed: " << e.what() << "\n";
    return 1;
  }
}
