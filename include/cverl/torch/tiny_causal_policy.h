#pragma once

#include <torch/torch.h>

#include <cstdint>

namespace cverl::torch_backend {

// Tiny causal language-model-style policy used for CPU smoke runs and unit
// tests. Architecture:
//   embedding[V, H] -> mean-pool over context -> Linear[H, H] -> ReLU
//   -> Linear[H, V]
//
// Forward consumes a (prompt_ids, response_ids) pair and returns logits for
// every response position, conditioned on the prompt + the prefix of the
// response. Despite being a bag-of-tokens "context" rather than a real
// causal transformer, it has the right input/output shape for the GRPO
// trainer and exercises the gradient path.
//
// On real GPU runs this is replaced by the production model wrapper. The
// trainer code only depends on the (logits over response positions) interface
// so swapping the backbone is local.
struct TinyCausalPolicyImpl : torch::nn::Module {
  TinyCausalPolicyImpl(int64_t vocab_size, int64_t hidden_dim, int32_t pad_id);

  // Returns logits with shape [B, R, V] over response positions, where the
  // i-th position is conditioned on prompt + response[<i] (excluding
  // padding).
  torch::Tensor forward(const torch::Tensor& prompt_ids, const torch::Tensor& response_ids);

  int64_t vocab_size_;
  int64_t hidden_dim_;
  int32_t pad_id_;

  torch::nn::Embedding embedding{nullptr};
  torch::nn::Linear hidden{nullptr};
  torch::nn::Linear lm_head{nullptr};
};

TORCH_MODULE(TinyCausalPolicy);

// Compute per-token log-probabilities of `response_ids` under `logits`.
//
// Inputs:
//   logits        float [B, R, V]
//   response_ids  long  [B, R]
// Output:
//   log_probs     float [B, R]
torch::Tensor response_log_probs(const torch::Tensor& logits, const torch::Tensor& response_ids);

}  // namespace cverl::torch_backend
