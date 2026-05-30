// Smoke test: rollout-driven GRPO step with the real Qwen3.5 policy.
//
// Mirrors tests/rollout/test_gsm8k_grpo_trainer.cc, but swaps the bag-of-tokens
// TinyCausalPolicy for Qwen3_5CausalLmPolicy on whichever device CUDA is
// available on. We mix correct and wrong samples per prompt so GRPO gives
// non-zero advantages and the PPO step has to actually move parameters.
//
// Skipped (returns 0) if no model dir is available so CPU CI runs that don't
// have Qwen3.5-0.8B on disk still pass `make test`.

#include "cverl/reward/gsm8k.h"
#include "cverl/rollout/rollout_batch.h"
#include "cverl/rollout/transport.h"
#include "cverl/text/byte_tokenizer.h"
#include "cverl/text/hf_bpe_tokenizer.h"
#include "cverl/torch/causal_lm_policy.h"
#include "cverl/torch/core_algos_torch.h"
#include "cverl/torch/tiny_causal_policy.h"  // for response_log_probs

#include <torch/torch.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool cond, const std::string& msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

cverl::rollout::RolloutResponse make_fake_response(const std::vector<std::string>& gold,
                                                   uint32_t n) {
  cverl::rollout::RolloutResponse resp;
  for (uint32_t p = 0; p < gold.size(); ++p) {
    for (uint32_t s = 0; s < n; ++s) {
      cverl::rollout::RolloutSequence seq;
      seq.prompt_index = p;
      seq.sample_index = s;
      // Half correct / half wrong per prompt so each group has within-group
      // variance and GRPO advantages are non-zero.
      if (s % 2 == 0) {
        seq.text = "Working...\n#### " + gold[p];
      } else {
        seq.text = "Working...\n#### 9999";
      }
      seq.finish_reason = "stop";
      resp.sequences.push_back(seq);
    }
  }
  return resp;
}

int64_t env_i64(const char* name, int64_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || std::string(value).empty()) {
    return fallback;
  }
  return std::strtoll(value, nullptr, 10);
}

torch::ScalarType env_dtype(const char* name, torch::ScalarType fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || std::string(value).empty()) {
    return fallback;
  }
  std::string dtype = value;
  if (dtype == "float32" || dtype == "fp32" || dtype == "f32") {
    return torch::kFloat32;
  }
  if (dtype == "bfloat16" || dtype == "bf16") {
    return torch::kBFloat16;
  }
  if (dtype == "float16" || dtype == "fp16" || dtype == "f16") {
    return torch::kFloat16;
  }
  throw std::invalid_argument("CVERL_QWEN_PARAM_DTYPE must be float32|bfloat16|float16");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string model_dir;
    if (argc > 1) {
      model_dir = argv[1];
    } else if (const char* env = std::getenv("CVERL_QWEN_MODEL_DIR")) {
      model_dir = env;
    } else {
      std::cout << "test_qwen3_5_grpo_step skipped: set CVERL_QWEN_MODEL_DIR "
                   "or pass model_dir as argv[1]\n";
      return 0;
    }
    auto tokenizer_json = std::filesystem::path(model_dir) / "tokenizer.json";
    if (!std::filesystem::exists(tokenizer_json)) {
      std::cout << "test_qwen3_5_grpo_step skipped: no tokenizer.json under "
                << model_dir << "\n";
      return 0;
    }

    torch::manual_seed(123);
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
      device = torch::Device(torch::kCUDA, 0);
    }

    std::vector<std::string> prompts = {
        "What is 1+1?",
        "What is 2+2?",
    };
    std::vector<std::string> gold = {"2", "4"};
    const uint32_t n = 4;

    auto resp = make_fake_response(gold, n);

    cverl::text::HfBpeTokenizerOptions tok_opts;
    tok_opts.tokenizer_json_path = tokenizer_json.string();
    cverl::text::HfBpeTokenizer tokenizer(tok_opts);

    cverl::reward::Gsm8kRewardOptions reward_opts;
    reward_opts.method = cverl::reward::Gsm8kExtractionMethod::Flexible;

    cverl::rollout::RolloutBatchOptions batch_opts;
    batch_opts.max_prompt_tokens = 32;
    batch_opts.max_response_tokens = 32;
    auto batch = cverl::rollout::build_gsm8k_rollout_batch(
        resp, prompts, gold, reward_opts, tokenizer, batch_opts);

    cverl::torch_backend::CausalLmPolicyOptions policy_opts;
    policy_opts.kind = cverl::torch_backend::CausalLmPolicyOptions::Kind::kQwen3_5;
    policy_opts.pad_id = tokenizer.pad_id() >= 0 ? tokenizer.pad_id() : 0;
    policy_opts.qwen_model_dir = model_dir;
    policy_opts.qwen_max_layers = env_i64("CVERL_QWEN_MAX_LAYERS", 2);
    policy_opts.param_dtype = env_dtype("CVERL_QWEN_PARAM_DTYPE", torch::kFloat32);
    auto policy = cverl::torch_backend::make_causal_lm_policy(policy_opts);
    policy->to_device(device);

    auto prompt_ids = batch.prompt_ids.to(device);
    auto response_ids = batch.response_ids.to(device);
    auto response_mask = batch.response_mask.to(device);
    auto token_rewards = batch.token_rewards.to(device);
    auto group_ids = batch.group_ids.to(device);

    torch::Tensor returns;
    auto advantages = cverl::torch_backend::grpo_outcome_advantage(
        token_rewards, response_mask, group_ids, 1.0e-6, true, &returns);
    auto adv_abs_sum = advantages.abs().sum().item<double>();
    require(adv_abs_sum > 0.0, "expected non-zero advantages from mixed-correctness samples");

    torch::Tensor old_log_probs;
    {
      torch::NoGradGuard no_grad;
      auto logits = policy->forward(prompt_ids, response_ids);
      old_log_probs = cverl::torch_backend::response_log_probs(logits, response_ids).detach();
      require(old_log_probs.isfinite().all().item<bool>(), "old_log_probs finite");
    }

    // Snapshot parameters before the optimizer step.
    std::vector<torch::Tensor> params_before;
    params_before.reserve(policy->parameters().size());
    {
      torch::NoGradGuard no_grad;
      for (const auto& p : policy->parameters()) {
        params_before.push_back(p.detach().clone());
      }
    }

    torch::optim::AdamW optimizer(
        policy->parameters(),
        torch::optim::AdamWOptions(3e-5).weight_decay(0.0));

    auto logits = policy->forward(prompt_ids, response_ids);
    auto log_probs = cverl::torch_backend::response_log_probs(logits, response_ids);
    auto loss = cverl::torch_backend::ppo_clipped_loss(
        old_log_probs, log_probs, advantages, response_mask,
        0.2, -1.0, -1.0, 3.0, CVERL_LOSS_AGG_TOKEN_MEAN);
    optimizer.zero_grad();
    loss.pg_loss.backward();
    optimizer.step();

    double loss_value = loss.pg_loss.item<double>();
    require(std::isfinite(loss_value), "pg_loss finite");

    // At least one parameter must have moved.
    double param_delta = 0.0;
    {
      torch::NoGradGuard no_grad;
      auto params_after = policy->parameters();
      for (size_t i = 0; i < params_before.size(); ++i) {
        param_delta += (params_after[i].detach() - params_before[i]).abs().sum().item<double>();
      }
    }
    require(param_delta > 0.0, "Qwen policy parameters must move after PPO step");

    std::cout << "test_qwen3_5_grpo_step passed (loss=" << loss_value
              << ", adv_abs_sum=" << adv_abs_sum
              << ", param_delta=" << param_delta
              << ", layers=" << policy_opts.qwen_max_layers
              << ", device=" << (device.is_cuda() ? "cuda" : "cpu") << ")\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_qwen3_5_grpo_step failed: " << e.what() << "\n";
    return 1;
  }
}
