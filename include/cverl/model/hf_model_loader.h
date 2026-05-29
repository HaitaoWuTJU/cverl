#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

namespace cverl {

struct HfConfigSummary {
  std::string model_type;
  std::string architecture;
  std::string dtype;
  int64_t vocab_size = 0;
  int64_t hidden_size = 0;
  int64_t num_hidden_layers = 0;
  int64_t num_attention_heads = 0;
  int64_t num_key_value_heads = 0;
};

struct SafetensorInfo {
  std::string name;
  std::string filename;
  std::string dtype;
  std::vector<int64_t> shape;
  uint64_t data_begin = 0;
  uint64_t data_end = 0;
};

class HfModelLoader {
 public:
  explicit HfModelLoader(std::string model_dir);

  void load_metadata();
  torch::Tensor load_tensor(const std::string& name) const;
  std::unordered_map<std::string, torch::Tensor> load_all_tensors() const;

  const std::string& model_dir() const { return model_dir_; }
  const HfConfigSummary& config() const { return config_; }
  const std::vector<SafetensorInfo>& tensors() const { return tensors_; }
  const SafetensorInfo* find_tensor(const std::string& name) const;

 private:
  std::string model_dir_;
  HfConfigSummary config_;
  std::vector<SafetensorInfo> tensors_;
  std::unordered_map<std::string, size_t> tensor_index_;
};

}  // namespace cverl
