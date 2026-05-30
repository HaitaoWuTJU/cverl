#pragma once

#include <memory>

#include "cverl/torch/causal_lm_policy.h"
#include "cverl/torch/tiny_causal_policy.h"

namespace cverl::torch_backend {

// Adapter that wraps the existing TinyCausalPolicy in the CausalLmPolicy
// interface. Used by CPU smoke tests / CI runs.
class TinyCausalLmPolicy : public CausalLmPolicy {
 public:
  TinyCausalLmPolicy(int64_t vocab_size, int64_t hidden_dim, int32_t pad_id);

  torch::Tensor forward(const torch::Tensor& prompt_ids,
                        const torch::Tensor& response_ids) override;

  int64_t vocab_size() const override { return vocab_size_; }
  int32_t pad_id() const override { return pad_id_; }

  std::shared_ptr<CausalLmPolicy> clone_as_reference() const override;

 private:
  int64_t vocab_size_;
  int64_t hidden_dim_;
  int32_t pad_id_;
  TinyCausalPolicy inner_;
};

}  // namespace cverl::torch_backend
