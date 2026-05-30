#pragma once

#include <memory>
#include <string>

#include <torch/torch.h>

namespace cverl::torch_backend {

// Abstract policy that the rollout-driven GRPO/PPO trainer talks to.
//
// All training code (advantage construction, PPO loss, optimizer step, KL
// penalty) is written against this interface so the backbone can be swapped
// between:
//   * TinyCausalLmPolicy   - 1-layer bag-of-tokens model for CPU smoke
//                            tests (vocab=ByteTokenizer or HF, hidden small).
//   * Qwen3_5CausalLmPolicy - real Qwen3.5-style causal LM that shares
//                              weights with the SGLang/vLLM rollout server.
//
// Contract:
//   forward(prompt_ids, response_ids) returns logits with shape
//   [B, R, V] where logits[b, t, :] are the next-token logits used to score
//   response_ids[b, t]. Implementations should mirror standard causal LMs:
//   the position predicting response_ids[b, t] sees prompt[b] followed by
//   response_ids[b, :t].
//
// The trainer treats prompt_ids as left-padded and response_ids as
// right-padded with `pad_id()`; padding columns must be tolerated but their
// logits at padded positions are masked out by `response_mask` upstream.
class CausalLmPolicy : public torch::nn::Module {
 public:
  ~CausalLmPolicy() override = default;

  // Returns logits with shape [B, R, V] (float32).
  virtual torch::Tensor forward(const torch::Tensor& prompt_ids,
                                const torch::Tensor& response_ids) = 0;

  // Vocabulary size used to size logits and embeddings.
  virtual int64_t vocab_size() const = 0;

  // Padding token id. Must match what the caller pre-pads with.
  virtual int32_t pad_id() const = 0;

  // Move every owned parameter / buffer to the given device.
  // Default impl walks parameters() and buffers().
  virtual void to_device(torch::Device device);

  // Snapshot every parameter into a frozen reference policy of the same
  // backbone. Used to compute KL penalty against the initial policy.
  virtual std::shared_ptr<CausalLmPolicy> clone_as_reference() const = 0;

  // Export the current policy in a Hugging Face checkpoint layout that
  // rollout engines can reload from disk. Backends that are not HF-style LMs
  // leave the default implementation unsupported.
  virtual bool supports_hf_checkpoint_export() const { return false; }
  virtual void save_hf_checkpoint(const std::string& output_dir,
                                  const std::string& dtype = "bfloat16") const;
};

// Factory descriptor for `make_causal_lm_policy`.
struct CausalLmPolicyOptions {
  enum class Kind {
    kTiny,
    kQwen3_5,
  };

  Kind kind = Kind::kTiny;

  // Common
  int32_t pad_id = 0;

  // Tiny backend
  int64_t tiny_vocab_size = 256;
  int64_t tiny_hidden_dim = 32;

  // Qwen3.5 backend
  std::string qwen_model_dir;
  // Truncate the layer stack for fast smoke runs. -1 keeps every layer.
  int64_t qwen_max_layers = -1;
};

std::shared_ptr<CausalLmPolicy> make_causal_lm_policy(const CausalLmPolicyOptions& options);

}  // namespace cverl::torch_backend
