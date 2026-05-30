#include "cverl/torch/causal_lm_policy.h"

#include <stdexcept>

#include "cverl/torch/qwen3_5_causal_lm_policy.h"
#include "cverl/torch/tiny_causal_lm_policy.h"

namespace cverl::torch_backend {

void CausalLmPolicy::to_device(torch::Device device) {
  // Default implementation: lean on nn::Module::to. Subclasses that hold
  // pointers to specific tensors outside the parameter list must override
  // this and re-publish those references after the base call.
  this->to(device);
}

void CausalLmPolicy::save_hf_checkpoint(const std::string&,
                                        const std::string&) const {
  throw std::runtime_error("this CausalLmPolicy backend cannot export HF checkpoints");
}

std::shared_ptr<CausalLmPolicy> make_causal_lm_policy(const CausalLmPolicyOptions& options) {
  switch (options.kind) {
    case CausalLmPolicyOptions::Kind::kTiny:
      return std::make_shared<TinyCausalLmPolicy>(
          options.tiny_vocab_size, options.tiny_hidden_dim, options.pad_id);
    case CausalLmPolicyOptions::Kind::kQwen3_5:
      if (options.qwen_model_dir.empty()) {
        throw std::invalid_argument("qwen3_5 policy requires qwen_model_dir");
      }
      return std::make_shared<Qwen3_5CausalLmPolicy>(
          options.qwen_model_dir, options.pad_id, options.qwen_max_layers, options.param_dtype);
  }
  throw std::invalid_argument("unsupported CausalLmPolicy kind");
}

}  // namespace cverl::torch_backend
