#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

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
  torch::Tensor forward_hidden(const torch::Tensor& input_ids, int64_t max_layers = -1);
  torch::Tensor forward_logits(const torch::Tensor& input_ids, int64_t max_layers = -1);

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
};

Qwen35TextConfig load_qwen35_text_config(const std::string& model_dir);

}  // namespace cverl
