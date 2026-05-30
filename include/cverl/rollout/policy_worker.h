#pragma once

#include <memory>
#include <vector>

#include "cverl/rollout/worker.h"
#include "cverl/torch/causal_lm_policy.h"

namespace cverl::rollout {

// GPU-resident rollout worker backed directly by a CausalLmPolicy.
//
// This is the native efficient worker used by cverl itself. It keeps prompt
// tokens, generated tokens, logits, and logprobs as torch::Tensor objects on
// the policy device. vLLM/Megatron integrations should match this contract at
// their worker/plugin boundary so the trainer does not fall back to HTTP,
// JSON, or checkpoint reloads.
class PolicyRolloutWorker final : public RolloutWorker {
 public:
  explicit PolicyRolloutWorker(std::shared_ptr<cverl::torch_backend::CausalLmPolicy> policy);

  GenerationOutput generate(const TokenBatch& prompts,
                            const GenerationConfig& config) override;

  std::vector<cverl::distributed::ParameterView> actor_parameters() override;

  cverl::torch_backend::CausalLmPolicy& policy() { return *policy_; }
  const cverl::torch_backend::CausalLmPolicy& policy() const { return *policy_; }

 private:
  std::shared_ptr<cverl::torch_backend::CausalLmPolicy> policy_;
};

}  // namespace cverl::rollout
