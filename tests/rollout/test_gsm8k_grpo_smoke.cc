// Smoke for the rollout-driven GRPO step:
// builds a fake rollout batch directly (no transport), runs one PPO update
// through TinyCausalPolicy, and asserts the loss is finite and at least one
// parameter changed.
#include "cverl/reward/gsm8k.h"
#include "cverl/rollout/rollout_batch.h"
#include "cverl/rollout/transport.h"
#include "cverl/text/byte_tokenizer.h"
#include "cverl/torch/core_algos_torch.h"
#include "cverl/torch/tiny_causal_policy.h"

#include <torch/torch.h>

#include <cmath>
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

cverl::rollout::RolloutResponse make_fake_response(const std::vector<std::string>& gold_per_prompt,
                                                   uint32_t n) {
  cverl::rollout::RolloutResponse resp;
  for (uint32_t p = 0; p < gold_per_prompt.size(); ++p) {
    for (uint32_t s = 0; s < n; ++s) {
      cverl::rollout::RolloutSequence seq;
      seq.prompt_index = p;
      seq.sample_index = s;
      // Half of the samples are correct so GRPO advantages are non-zero.
      if (s % 2 == 0) {
        seq.text = "Correct working out\n#### " + gold_per_prompt[p];
      } else {
        seq.text = "Wrong working out\n#### 9999";
      }
      seq.finish_reason = "stop";
      resp.sequences.push_back(seq);
    }
  }
  return resp;
}

}  // namespace

int main() {
  try {
    torch::manual_seed(123);

    std::vector<std::string> prompts = {
        "What is 1+1?",
        "What is 2+2?",
        "What is 3+5?",
        "What is 7-2?",
    };
    std::vector<std::string> gold = {"2", "4", "8", "5"};
    const uint32_t n = 4;

    auto resp = make_fake_response(gold, n);

    cverl::text::ByteTokenizer tokenizer;
    cverl::reward::Gsm8kRewardOptions reward_opts;
    cverl::rollout::RolloutBatchOptions batch_opts;
    batch_opts.max_prompt_tokens = 32;
    batch_opts.max_response_tokens = 48;

    auto batch = cverl::rollout::build_gsm8k_rollout_batch(
        resp, prompts, gold, reward_opts, tokenizer, batch_opts);

    const int64_t hidden_dim = 16;
    cverl::torch_backend::TinyCausalPolicy policy(
        cverl::text::ByteTokenizer::vocab_size(),
        hidden_dim,
        cverl::text::ByteTokenizer::pad_id());
    torch::optim::AdamW optimizer(policy->parameters(), torch::optim::AdamWOptions(1e-2));

    // Capture initial weights for the parameter-update check.
    auto initial_lm_head = policy->lm_head->weight.detach().clone();

    torch::Tensor returns;
    torch::Tensor advantages = cverl::torch_backend::grpo_outcome_advantage(
        batch.token_rewards, batch.response_mask, batch.group_ids, 1.0e-6, true, &returns);

    torch::Tensor old_log_probs;
    {
      torch::NoGradGuard no_grad;
      torch::Tensor logits = policy->forward(batch.prompt_ids, batch.response_ids);
      old_log_probs = cverl::torch_backend::response_log_probs(logits, batch.response_ids).detach();
      // Sanity: log_probs are <= 0 and finite.
      require(old_log_probs.isfinite().all().item<bool>(), "old_log_probs finite");
      require((old_log_probs <= 1e-4).all().item<bool>(), "old_log_probs non-positive");
      // Shape matches response_ids.
      require(old_log_probs.sizes() == batch.response_ids.sizes(), "old_log_probs shape");
    }

    // One PPO epoch.
    torch::Tensor logits = policy->forward(batch.prompt_ids, batch.response_ids);
    torch::Tensor log_probs = cverl::torch_backend::response_log_probs(logits, batch.response_ids);
    auto loss = cverl::torch_backend::ppo_clipped_loss(
        old_log_probs, log_probs, advantages, batch.response_mask,
        0.2, -1.0, -1.0, 3.0, CVERL_LOSS_AGG_TOKEN_MEAN);
    optimizer.zero_grad();
    loss.pg_loss.backward();
    optimizer.step();

    double loss_value = loss.pg_loss.item<double>();
    require(std::isfinite(loss_value), "loss is finite");
    // The first PPO epoch with old == new logprob should give pg_loss ~ -E[adv].
    // We don't pin the exact value; just check that something changed.

    // At least one optimizer step actually moved a parameter.
    auto delta = (policy->lm_head->weight.detach() - initial_lm_head).abs().sum().item<double>();
    require(delta > 0.0, "policy parameters updated");

    std::cout << "gsm8k grpo smoke tests passed (loss=" << loss_value << ", param_delta=" << delta << ")\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_gsm8k_grpo_smoke failed: " << e.what() << "\n";
    return 1;
  }
}
