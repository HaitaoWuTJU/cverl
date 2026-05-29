#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

#include "cverl/reward/gsm8k.h"
#include "cverl/rollout/transport.h"
#include "cverl/text/tokenizer.h"

namespace cverl::rollout {

// Tokenized, padded view of a RolloutResponse ready to feed the GRPO/PPO
// trainer. All tensors live on CPU; callers move them to the right device.
//
// Shapes (B = sequences = prompts.size() * n, P = max prompt len, R = max
// response len):
//   prompt_ids       int64 [B, P]  pad = pad_id
//   response_ids     int64 [B, R]  pad = pad_id
//   response_mask  float32 [B, R]  1 where the position is a real response
//                                  token, 0 on padding / past EOS
//   group_ids        int64 [B]     prompt_index for each row
//   scalar_rewards float32 [B]     rule reward per sequence
//   token_rewards  float32 [B, R]  scalar reward placed on the last real
//                                  response token, 0 elsewhere
struct RolloutBatch {
  torch::Tensor prompt_ids;
  torch::Tensor response_ids;
  torch::Tensor response_mask;
  torch::Tensor group_ids;
  torch::Tensor scalar_rewards;
  torch::Tensor token_rewards;
  // The original response strings, kept around for logging / debugging.
  std::vector<std::string> response_texts;
};

struct RolloutBatchOptions {
  int64_t max_prompt_tokens = 256;
  int64_t max_response_tokens = 256;
  bool add_bos = false;
  bool add_eos = true;
};

// Build a tokenized batch from a transport response.
//
// `prompts[i]` and `ground_truths[i]` correspond to prompt index i; the
// transport response is expected to contain `prompts.size() * n` sequences in
// (prompt_index, sample_index) order. Rewards are computed via
// compute_gsm8k_reward(seq.text, ground_truths[seq.prompt_index], ...).
RolloutBatch build_gsm8k_rollout_batch(const RolloutResponse& response,
                                       const std::vector<std::string>& prompts,
                                       const std::vector<std::string>& ground_truths,
                                       const cverl::reward::Gsm8kRewardOptions& reward_options,
                                       const cverl::text::Tokenizer& tokenizer,
                                       const RolloutBatchOptions& options = {});

}  // namespace cverl::rollout
