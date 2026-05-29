#include "cverl/torch/tiny_causal_policy.h"

#include <stdexcept>

namespace cverl::torch_backend {

TinyCausalPolicyImpl::TinyCausalPolicyImpl(int64_t vocab_size, int64_t hidden_dim, int32_t pad_id)
    : vocab_size_(vocab_size), hidden_dim_(hidden_dim), pad_id_(pad_id) {
  if (vocab_size <= 1 || hidden_dim <= 0) {
    throw std::invalid_argument("TinyCausalPolicy: invalid dims");
  }
  embedding = register_module(
      "embedding",
      torch::nn::Embedding(torch::nn::EmbeddingOptions(vocab_size, hidden_dim).padding_idx(pad_id)));
  hidden = register_module("hidden", torch::nn::Linear(hidden_dim, hidden_dim));
  lm_head = register_module("lm_head", torch::nn::Linear(hidden_dim, vocab_size));
}

torch::Tensor TinyCausalPolicyImpl::forward(const torch::Tensor& prompt_ids,
                                            const torch::Tensor& response_ids) {
  TORCH_CHECK(prompt_ids.dim() == 2, "prompt_ids must be [B, P]");
  TORCH_CHECK(response_ids.dim() == 2, "response_ids must be [B, R]");
  TORCH_CHECK(prompt_ids.size(0) == response_ids.size(0), "batch mismatch");

  const int64_t B = response_ids.size(0);
  const int64_t R = response_ids.size(1);

  // Embed prompt + response and build a non-padding mask used for averaging.
  torch::Tensor prompt_mask = (prompt_ids != pad_id_).to(torch::kFloat32);  // [B, P]
  torch::Tensor response_mask = (response_ids != pad_id_).to(torch::kFloat32);  // [B, R]

  torch::Tensor prompt_emb = embedding(prompt_ids);  // [B, P, H]
  torch::Tensor response_emb = embedding(response_ids);  // [B, R, H]

  // Sum over prompt tokens (mask out padding).
  torch::Tensor prompt_sum = (prompt_emb * prompt_mask.unsqueeze(-1)).sum(1);  // [B, H]
  torch::Tensor prompt_count = prompt_mask.sum(1).clamp_min(1.0);  // [B]

  // Cumulative response context: at position i we want the average of
  // prompt + response[<i] (excluding padding). Use cumulative sum so this is
  // O(BR) instead of O(BR^2).
  torch::Tensor response_masked = response_emb * response_mask.unsqueeze(-1);
  torch::Tensor cumsum_resp = response_masked.cumsum(1);  // [B, R, H]
  torch::Tensor cumcount_resp = response_mask.cumsum(1);  // [B, R]

  // At position i the prefix is (prompt) + (response[0..i-1]). Shift cumsum
  // right by one so position 0 sees only the prompt.
  torch::Tensor zero_resp = torch::zeros({B, 1, hidden_dim_}, response_emb.options());
  torch::Tensor zero_count = torch::zeros({B, 1}, response_mask.options());
  torch::Tensor prefix_sum = torch::cat({zero_resp, cumsum_resp.slice(1, 0, R - 1)}, 1);  // [B, R, H]
  torch::Tensor prefix_count = torch::cat({zero_count, cumcount_resp.slice(1, 0, R - 1)}, 1);  // [B, R]

  torch::Tensor total_sum = prefix_sum + prompt_sum.unsqueeze(1);
  torch::Tensor total_count = (prefix_count + prompt_count.unsqueeze(1)).clamp_min(1.0);

  torch::Tensor context = total_sum / total_count.unsqueeze(-1);  // [B, R, H]

  torch::Tensor h = torch::relu(hidden(context));
  return lm_head(h);  // [B, R, V]
}

torch::Tensor response_log_probs(const torch::Tensor& logits, const torch::Tensor& response_ids) {
  TORCH_CHECK(logits.dim() == 3, "logits must be [B, R, V]");
  TORCH_CHECK(response_ids.dim() == 2, "response_ids must be [B, R]");
  TORCH_CHECK(logits.size(0) == response_ids.size(0), "batch mismatch");
  TORCH_CHECK(logits.size(1) == response_ids.size(1), "seq mismatch");
  return torch::log_softmax(logits, -1).gather(-1, response_ids.unsqueeze(-1)).squeeze(-1);
}

TinyCausalPolicy clone_as_reference(const TinyCausalPolicy& source) {
  TinyCausalPolicy ref(source->vocab_size_, source->hidden_dim_, source->pad_id_);
  // Copy parameters in their current state, then freeze.
  torch::NoGradGuard no_grad;
  auto src_params = source->named_parameters(/*recurse=*/true);
  auto ref_params = ref->named_parameters(/*recurse=*/true);
  for (const auto& kv : src_params) {
    auto& dst = ref_params[kv.key()];
    TORCH_CHECK(dst.defined(), "missing parameter in reference: " + kv.key());
    TORCH_CHECK(dst.sizes() == kv.value().sizes(),
                "parameter shape mismatch: " + kv.key());
    dst.copy_(kv.value());
  }
  // Buffers (none today, but be defensive against future additions).
  auto src_buffers = source->named_buffers(/*recurse=*/true);
  auto ref_buffers = ref->named_buffers(/*recurse=*/true);
  for (const auto& kv : src_buffers) {
    auto& dst = ref_buffers[kv.key()];
    if (dst.defined()) {
      dst.copy_(kv.value());
    }
  }
  for (auto& p : ref->parameters()) {
    p.set_requires_grad(false);
  }
  ref->eval();
  return ref;
}

}  // namespace cverl::torch_backend
