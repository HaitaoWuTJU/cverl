#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "cverl/distributed/parallel_ops.h"
#include "cverl/model/hf_model_loader.h"

namespace cverl {

struct Qwen35TextConfig {
  int64_t vocab_size = 248320;
  int64_t hidden_size = 1024;
  int64_t num_hidden_layers = 24;
  int64_t num_attention_heads = 8;
  int64_t num_key_value_heads = 2;
  int64_t head_dim = 256;
  int64_t intermediate_size = 3584;
  int64_t linear_key_head_dim = 128;
  int64_t linear_value_head_dim = 128;
  int64_t linear_num_key_heads = 16;
  int64_t linear_num_value_heads = 16;
  int64_t linear_conv_kernel_dim = 4;
  double rms_norm_eps = 1e-6;
  double rope_theta = 10000000.0;
  double partial_rotary_factor = 0.25;
  std::vector<std::string> layer_types;
};

class Qwen35TextModel {
 public:
  explicit Qwen35TextModel(HfModelLoader loader);

  const Qwen35TextConfig& config() const { return config_; }
  const HfModelLoader& loader() const { return loader_; }
  void to(torch::Device device);
  torch::Tensor token_embeddings(const torch::Tensor& input_ids);
  torch::Tensor forward_hidden(const torch::Tensor& input_ids, int64_t max_layers = -1);
  torch::Tensor forward_logits(const torch::Tensor& input_ids, int64_t max_layers = -1);
  torch::Tensor forward_hidden_tensor_parallel(const torch::Tensor& input_ids,
                                               const distributed::ParallelGroup& tensor_group,
                                               int64_t max_layers = -1);
  torch::Tensor forward_hidden_range_tensor_parallel(const torch::Tensor& hidden,
                                                     int64_t layer_begin,
                                                     int64_t layer_end,
                                                     const distributed::ParallelGroup& tensor_group,
                                                     bool apply_final_norm);
  torch::Tensor mlp_tensor_parallel(const torch::Tensor& x,
                                    int64_t layer_idx,
                                    const distributed::ParallelGroup& tensor_group);
  torch::Tensor full_attention_tensor_parallel(const torch::Tensor& x,
                                               int64_t layer_idx,
                                               const distributed::ParallelGroup& tensor_group);
  torch::Tensor linear_attention_tensor_parallel(const torch::Tensor& x,
                                                 int64_t layer_idx,
                                                 const distributed::ParallelGroup& tensor_group);

  // List every tensor name that the forward path is expected to look up,
  // for `max_layers` first transformer blocks (or all layers if -1). Useful
  // for higher-level wrappers (CausalLmPolicy etc) that need to materialize
  // every weight up-front and attach autograd / optimizer state to them.
  std::vector<std::string> required_weight_names(int64_t max_layers = -1) const;

  // Replace the lazy safetensors-backed weight cache for the given name with
  // a caller-owned tensor. Subsequent forward calls will use this tensor
  // directly (no copy). This is how `Qwen3_5CausalLmPolicy` injects its
  // registered nn::Module parameters into the forward graph so backward()
  // reaches them.
  void set_weight_override(const std::string& name, torch::Tensor tensor);

 private:
  torch::Tensor weight(const std::string& name);
  torch::Tensor embed(const torch::Tensor& input_ids);
  torch::Tensor rms_norm(const torch::Tensor& x, const torch::Tensor& weight) const;
  torch::Tensor rms_norm_gated(const torch::Tensor& x, const torch::Tensor& gate, const torch::Tensor& weight) const;
  torch::Tensor mlp(const torch::Tensor& x, int64_t layer_idx);
  torch::Tensor full_attention(const torch::Tensor& x, int64_t layer_idx);
  torch::Tensor linear_attention(const torch::Tensor& x, int64_t layer_idx);
  std::pair<torch::Tensor, torch::Tensor> rotary_embeddings(int64_t batch_size,
                                                            int64_t seq_len,
                                                            torch::Device device,
                                                            torch::ScalarType dtype) const;

  HfModelLoader loader_;
  Qwen35TextConfig config_;
  std::unordered_map<std::string, torch::Tensor> weights_;
  torch::Device device_ = torch::kCPU;
};

Qwen35TextConfig load_qwen35_text_config(const std::string& model_dir);

}  // namespace cverl
