#include "cverl/torch/tiny_causal_lm_policy.h"

namespace cverl::torch_backend {

TinyCausalLmPolicy::TinyCausalLmPolicy(int64_t vocab_size,
                                       int64_t hidden_dim,
                                       int32_t pad_id)
    : vocab_size_(vocab_size),
      hidden_dim_(hidden_dim),
      pad_id_(pad_id),
      inner_(vocab_size, hidden_dim, pad_id) {
  // Surface the inner Module's parameters / submodules through this Module so
  // optimizer construction (`policy->parameters()`) sees them.
  register_module("inner", inner_);
}

torch::Tensor TinyCausalLmPolicy::forward(const torch::Tensor& prompt_ids,
                                          const torch::Tensor& response_ids) {
  return inner_->forward(prompt_ids, response_ids);
}

std::shared_ptr<CausalLmPolicy> TinyCausalLmPolicy::clone_as_reference() const {
  auto ref = std::make_shared<TinyCausalLmPolicy>(vocab_size_, hidden_dim_, pad_id_);
  // Copy the inner policy parameters and freeze.
  torch::NoGradGuard no_grad;
  auto src_named = const_cast<TinyCausalPolicyImpl&>(*inner_).named_parameters(true);
  auto dst_named = ref->inner_->named_parameters(true);
  for (const auto& kv : src_named) {
    auto& dst = dst_named[kv.key()];
    dst.copy_(kv.value());
  }
  for (auto& p : ref->inner_->parameters()) {
    p.set_requires_grad(false);
  }
  ref->inner_->eval();
  return ref;
}

}  // namespace cverl::torch_backend
