#include "cverl/rollout/rollout_batch.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace cverl::rollout {

namespace {

void left_pad_into_row(const std::vector<int32_t>& ids,
                       int64_t* dst_ptr,
                       int64_t row_len,
                       int32_t pad_id) {
  // Right-aligned padding so the prompt's last token is always at column R-1.
  int64_t take = std::min<int64_t>(static_cast<int64_t>(ids.size()), row_len);
  int64_t leading_pad = row_len - take;
  for (int64_t c = 0; c < leading_pad; ++c) {
    dst_ptr[c] = pad_id;
  }
  for (int64_t c = 0; c < take; ++c) {
    dst_ptr[leading_pad + c] = static_cast<int64_t>(ids[static_cast<size_t>(ids.size()) - take + c]);
  }
}

void right_pad_into_row(const std::vector<int32_t>& ids,
                        int64_t* dst_ptr,
                        int64_t row_len,
                        int32_t pad_id) {
  int64_t take = std::min<int64_t>(static_cast<int64_t>(ids.size()), row_len);
  for (int64_t c = 0; c < take; ++c) {
    dst_ptr[c] = static_cast<int64_t>(ids[static_cast<size_t>(c)]);
  }
  for (int64_t c = take; c < row_len; ++c) {
    dst_ptr[c] = pad_id;
  }
}

}  // namespace

RolloutBatch build_gsm8k_rollout_batch(const RolloutResponse& response,
                                       const std::vector<std::string>& prompts,
                                       const std::vector<std::string>& ground_truths,
                                       const cverl::reward::Gsm8kRewardOptions& reward_options,
                                       const cverl::text::Tokenizer& tokenizer,
                                       const RolloutBatchOptions& options) {
  if (prompts.size() != ground_truths.size()) {
    throw std::invalid_argument("prompts and ground_truths size mismatch");
  }
  if (options.max_prompt_tokens <= 0 || options.max_response_tokens <= 0) {
    throw std::invalid_argument("max_*_tokens must be positive");
  }

  const int64_t batch = static_cast<int64_t>(response.sequences.size());
  if (batch == 0) {
    throw std::invalid_argument("rollout response has no sequences");
  }
  const int32_t pad_id = tokenizer.pad_id();

  RolloutBatch out;
  auto long_options = torch::TensorOptions().dtype(torch::kInt64);
  auto float_options = torch::TensorOptions().dtype(torch::kFloat32);

  out.prompt_ids = torch::empty({batch, options.max_prompt_tokens}, long_options);
  out.response_ids = torch::empty({batch, options.max_response_tokens}, long_options);
  out.response_mask = torch::zeros({batch, options.max_response_tokens}, float_options);
  out.group_ids = torch::empty({batch}, long_options);
  out.scalar_rewards = torch::empty({batch}, float_options);
  out.token_rewards = torch::zeros({batch, options.max_response_tokens}, float_options);
  out.response_texts.reserve(static_cast<size_t>(batch));

  auto* prompt_ptr = out.prompt_ids.data_ptr<int64_t>();
  auto* response_ptr = out.response_ids.data_ptr<int64_t>();
  auto* response_mask_ptr = out.response_mask.data_ptr<float>();
  auto* group_ptr = out.group_ids.data_ptr<int64_t>();
  auto* scalar_ptr = out.scalar_rewards.data_ptr<float>();
  auto* token_reward_ptr = out.token_rewards.data_ptr<float>();

  cverl::text::EncodeOptions prompt_encode;
  prompt_encode.add_bos = options.add_bos;
  prompt_encode.add_eos = false;
  prompt_encode.max_tokens = static_cast<int32_t>(options.max_prompt_tokens);

  cverl::text::EncodeOptions response_encode;
  response_encode.add_bos = false;
  response_encode.add_eos = options.add_eos;
  response_encode.max_tokens = static_cast<int32_t>(options.max_response_tokens);

  // Cache prompt encodings since the same prompt is repeated n times in the
  // transport output.
  std::vector<std::vector<int32_t>> prompt_token_cache(prompts.size());
  std::vector<bool> prompt_cached(prompts.size(), false);

  for (int64_t i = 0; i < batch; ++i) {
    const auto& seq = response.sequences[static_cast<size_t>(i)];
    if (seq.prompt_index >= prompts.size()) {
      throw std::out_of_range("rollout sequence prompt_index out of range");
    }
    const size_t pidx = seq.prompt_index;
    if (!prompt_cached[pidx]) {
      prompt_token_cache[pidx] = tokenizer.encode(prompts[pidx], prompt_encode);
      prompt_cached[pidx] = true;
    }
    left_pad_into_row(prompt_token_cache[pidx],
                      prompt_ptr + i * options.max_prompt_tokens,
                      options.max_prompt_tokens,
                      pad_id);

    std::vector<int32_t> encoded_response_ids;
    const std::vector<int32_t>* response_ids = &seq.token_ids;
    if (response_ids->empty()) {
      encoded_response_ids = tokenizer.encode(seq.text, response_encode);
      response_ids = &encoded_response_ids;
    }
    int64_t real_len = std::min<int64_t>(static_cast<int64_t>(response_ids->size()), options.max_response_tokens);
    right_pad_into_row(*response_ids,
                       response_ptr + i * options.max_response_tokens,
                       options.max_response_tokens,
                       pad_id);
    for (int64_t c = 0; c < real_len; ++c) {
      response_mask_ptr[i * options.max_response_tokens + c] = 1.0f;
    }

    group_ptr[i] = static_cast<int64_t>(seq.prompt_index);

    double reward = cverl::reward::compute_gsm8k_reward(
        seq.text, ground_truths[seq.prompt_index], reward_options);
    scalar_ptr[i] = static_cast<float>(reward);
    if (real_len > 0) {
      token_reward_ptr[i * options.max_response_tokens + (real_len - 1)] = static_cast<float>(reward);
    }
    out.response_texts.push_back(seq.text);
  }
  return out;
}

}  // namespace cverl::rollout
