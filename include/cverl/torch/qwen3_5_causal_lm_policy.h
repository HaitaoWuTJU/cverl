#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cverl/model/qwen3_5_text.h"
#include "cverl/torch/causal_lm_policy.h"

namespace cverl::torch_backend {

// Causal-LM policy backed by `cverl::Qwen35TextModel`.
//
// On construction it loads every weight the forward path needs (up to
// `max_layers` transformer blocks) from the safetensors directory, registers
// them as fp32 trainable nn::Module parameters, and pushes the same tensors
// into the underlying `Qwen35TextModel` via `set_weight_override` so that
// `forward_logits` autograd-flows back through the registered parameters.
//
// Trainer-facing forward signature:
//   forward(prompt_ids[B,P], response_ids[B,R]) -> logits[B,R,V]
// The two inputs are concatenated along seq, run through `forward_logits`,
// and the response slice [P-1, P+R-1) is returned. This matches the standard
// causal-LM convention (the logits at position t score response_ids[t]) and
// has the same shape as TinyCausalLmPolicy::forward.
class Qwen3_5CausalLmPolicy : public CausalLmPolicy {
 public:
  Qwen3_5CausalLmPolicy(const std::string& model_dir,
                        int32_t pad_id,
                        int64_t max_layers = -1);

  torch::Tensor forward(const torch::Tensor& prompt_ids,
                        const torch::Tensor& response_ids) override;

  int64_t vocab_size() const override { return config_.vocab_size; }
  int32_t pad_id() const override { return pad_id_; }
  int64_t hidden_size() const { return config_.hidden_size; }

  void to_device(torch::Device device) override;
  std::shared_ptr<CausalLmPolicy> clone_as_reference() const override;

  const Qwen35TextConfig& config() const { return config_; }
  int64_t max_layers() const { return max_layers_; }

 private:
  // Returns the canonicalized parameter name registered with this Module.
  static std::string sanitize_name(const std::string& weight_name);

  // Re-attach every parameter from `params_` into `model_` via
  // set_weight_override. Called after construction and after to_device.
  void publish_overrides();

  std::string model_dir_;
  int32_t pad_id_;
  int64_t max_layers_;
  Qwen35TextConfig config_;

  std::unique_ptr<Qwen35TextModel> model_;

  // Original weight names <-> registered parameter names. nn::Module rejects
  // names containing '.', so we substitute '/' on registration and keep the
  // mapping for set_weight_override.
  std::vector<std::string> weight_names_;
  std::vector<std::string> registered_names_;
  std::unordered_map<std::string, torch::Tensor> params_;
};

}  // namespace cverl::torch_backend
