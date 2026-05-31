#include "cverl/model/hf_model_loader.h"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  std::string model_dir;
  if (argc == 2) {
    model_dir = argv[1];
  } else if (const char* env = std::getenv("CVERL_QWEN_MODEL_DIR")) {
    model_dir = env;
  } else {
    std::cout << "test_hf_model_loader skipped: set CVERL_QWEN_MODEL_DIR "
                 "or pass model_dir as argv[1]\n";
    return 0;
  }

  cverl::HfModelLoader loader(model_dir);
  loader.load_metadata();
  if (loader.config().model_type.empty() || loader.config().hidden_size <= 0 || loader.config().vocab_size <= 0) {
    std::cerr << "invalid config summary\n";
    return 1;
  }
  if (loader.tensors().empty()) {
    std::cerr << "no tensors found\n";
    return 1;
  }

  const auto& first = loader.tensors().front();
  torch::Tensor tensor = loader.load_tensor(first.name);
  if (tensor.numel() == 0 || tensor.sizes().size() != static_cast<int64_t>(first.shape.size())) {
    std::cerr << "loaded tensor shape invalid\n";
    return 1;
  }

  std::cout << "HF model loader test passed\n";
  std::cout << "model_type=" << loader.config().model_type << "\n";
  std::cout << "tensor_count=" << loader.tensors().size() << "\n";
  std::cout << "loaded_tensor=" << first.name << "\n";
  return 0;
}
