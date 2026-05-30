#include "cverl/rollout/policy_worker.h"

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "cverl/torch/causal_lm_policy.h"

namespace {

void require(bool cond, const std::string& msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

}  // namespace

int main() {
  try {
    std::string model_dir;
    if (const char* env = std::getenv("CVERL_QWEN_MODEL_DIR")) {
      model_dir = env;
    } else {
      model_dir = "/apdcephfs_fsgm3/share_305110755/hunyuan/marvinhtwu/rl/models/Qwen3.5-0.8B";
    }
    if (!torch::cuda::is_available()) {
      std::cout << "test_policy_rollout_worker skipped: CUDA is required for "
                   "Qwen3.5-0.8B rollout worker validation\n";
      return 0;
    }
    if (!std::filesystem::exists(std::filesystem::path(model_dir) / "config.json")) {
      std::cout << "test_policy_rollout_worker skipped: Qwen3.5-0.8B not found at "
                << model_dir << "\n";
      return 0;
    }

    cverl::torch_backend::CausalLmPolicyOptions opts;
    opts.kind = cverl::torch_backend::CausalLmPolicyOptions::Kind::kQwen3_5;
    opts.qwen_model_dir = model_dir;
    opts.qwen_max_layers = -1;
    opts.pad_id = 0;
    auto policy = cverl::torch_backend::make_causal_lm_policy(opts);
    policy->to_device(torch::Device(torch::kCUDA, 0));

    cverl::rollout::PolicyRolloutWorker worker(policy);
    auto params = worker.actor_parameters();
    require(!params.empty(), "worker exposes actor parameters");

    cverl::rollout::TokenBatch prompts;
    prompts.token_ids = torch::tensor({{1, 2, 3, 4}, {5, 6, 7, 8}}, torch::kLong);

    cverl::rollout::GenerationConfig config;
    config.max_new_tokens = 2;
    config.temperature = 0.0;
    config.eos_token_id = -1;
    auto out = worker.generate(prompts, config);

    require(out.token_ids.dim() == 2 && out.token_ids.size(0) == 2 && out.token_ids.size(1) == 2,
            "token shape");
    require(out.logprobs.dim() == 2 && out.logprobs.size(0) == 2 && out.logprobs.size(1) == 2,
            "logprob shape");
    require(out.lengths.dim() == 1 && out.lengths.size(0) == 2, "length shape");
    require(out.token_ids.scalar_type() == torch::kLong, "token dtype");
    require(out.logprobs.scalar_type() == torch::kFloat32, "logprob dtype");
    require(torch::all(out.lengths == 2).item<bool>(), "lengths without eos");
    require(out.token_ids.device() == policy->parameters().front().device(), "tokens stay on policy device");
    require(out.logprobs.device() == policy->parameters().front().device(), "logprobs stay on policy device");

    cverl::distributed::SingleProcessCollectives collectives;
    cverl::rollout::synchronize_rollout_actor_weights(worker, collectives, 0, {0});

    std::cout << "policy rollout worker Qwen3.5 tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_policy_rollout_worker failed: " << e.what() << "\n";
    return 1;
  }
}
