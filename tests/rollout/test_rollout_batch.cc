#include "cverl/reward/gsm8k.h"
#include "cverl/rollout/rollout_batch.h"
#include "cverl/rollout/transport.h"
#include "cverl/text/byte_tokenizer.h"

#include <torch/torch.h>

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

}  // namespace

int main() {
  try {
    // Build a fake rollout response: 2 prompts, n=3 samples each. Mark the
    // even-indexed samples as the gold answer to exercise reward / mask.
    cverl::rollout::RolloutResponse resp;
    std::vector<std::string> prompts = {"What is 1+1?", "What is 2+2?"};
    std::vector<std::string> ground_truths = {"2", "4"};
    const uint32_t n = 3;
    for (uint32_t p = 0; p < prompts.size(); ++p) {
      for (uint32_t s = 0; s < n; ++s) {
        cverl::rollout::RolloutSequence seq;
        seq.prompt_index = p;
        seq.sample_index = s;
        if (s == 0) {
          // Correct answer.
          seq.text = "The answer is\n#### " + ground_truths[p];
        } else if (s == 1) {
          // Wrong but parseable.
          seq.text = "wrong answer\n#### 999";
        } else {
          // No #### marker -> no_answer reward.
          seq.text = "rambling without structure";
        }
        seq.finish_reason = "stop";
        resp.sequences.push_back(seq);
      }
    }

    cverl::text::ByteTokenizer tok;
    cverl::reward::Gsm8kRewardOptions reward_opts;
    cverl::rollout::RolloutBatchOptions opts;
    opts.max_prompt_tokens = 32;
    opts.max_response_tokens = 48;
    opts.add_bos = false;
    opts.add_eos = true;

    auto batch = cverl::rollout::build_gsm8k_rollout_batch(
        resp, prompts, ground_truths, reward_opts, tok, opts);

    const int64_t B = static_cast<int64_t>(prompts.size()) * n;
    require(batch.prompt_ids.sizes() == torch::IntArrayRef({B, 32}), "prompt shape");
    require(batch.response_ids.sizes() == torch::IntArrayRef({B, 48}), "response shape");
    require(batch.response_mask.sizes() == torch::IntArrayRef({B, 48}), "mask shape");
    require(batch.scalar_rewards.sizes() == torch::IntArrayRef({B}), "rewards shape");
    require(batch.token_rewards.sizes() == torch::IntArrayRef({B, 48}), "token rewards shape");
    require(batch.group_ids.sizes() == torch::IntArrayRef({B}), "group ids shape");
    require(static_cast<int64_t>(batch.response_texts.size()) == B, "response texts size");

    // group_ids == prompt_index for each row.
    auto groups = batch.group_ids.to(torch::kInt64);
    for (uint32_t p = 0; p < prompts.size(); ++p) {
      for (uint32_t s = 0; s < n; ++s) {
        int64_t row = p * n + s;
        require(groups[row].item<int64_t>() == p, "group id matches prompt index");
      }
    }

    // Reward routing: sample 0 -> 1.0, sample 1 -> 0.0 (parsed wrong),
    // sample 2 -> 0.0 (no answer), and the mask sums match the encoded
    // response length.
    for (uint32_t p = 0; p < prompts.size(); ++p) {
      for (uint32_t s = 0; s < n; ++s) {
        int64_t row = p * n + s;
        double r = batch.scalar_rewards[row].item<double>();
        if (s == 0) {
          require(r == 1.0, "correct sample reward");
        } else {
          require(r == 0.0, "wrong/no-answer reward");
        }
        // Token reward fires only on the last real response token.
        double row_sum = batch.token_rewards[row].sum().item<double>();
        require(std::abs(row_sum - r) < 1e-9, "token reward sums to scalar reward");
      }
    }

    // Left-padded prompts: the last column is always a real token, not pad.
    auto last_col = batch.prompt_ids.select(1, opts.max_prompt_tokens - 1);
    auto pad_id = cverl::text::ByteTokenizer::pad_id();
    for (int64_t i = 0; i < B; ++i) {
      require(last_col[i].item<int64_t>() != pad_id, "last prompt col is real token");
    }

    // Right-padded responses: response_mask should equal (response_ids != pad)
    // up to the encoded length, then zeros.
    auto response_pad_match = (batch.response_ids != pad_id).to(torch::kFloat32);
    auto diff = (response_pad_match - batch.response_mask).abs().sum().item<double>();
    require(diff < 1e-6, "response mask matches non-pad positions");

    std::cout << "rollout batch tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_rollout_batch failed: " << e.what() << "\n";
    return 1;
  }
}
