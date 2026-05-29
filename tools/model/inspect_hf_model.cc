#include "cverl/model/hf_model_loader.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void usage(const char* argv0) {
  std::cerr << "usage: " << argv0 << " <model-dir> [--find SUBSTR] [--load-tensor NAME] [--load-all]\n";
}

std::string shape_string(const std::vector<int64_t>& shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      out += ",";
    }
    out += std::to_string(shape[i]);
  }
  out += "]";
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage(argv[0]);
    return 2;
  }

  std::string model_dir = argv[1];
  std::string find_substr;
  std::string load_tensor;
  bool load_all = false;
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--find") {
      if (i + 1 >= argc) {
        std::cerr << "--find requires a value\n";
        return 2;
      }
      find_substr = argv[++i];
    } else if (arg == "--load-tensor") {
      if (i + 1 >= argc) {
        std::cerr << "--load-tensor requires a value\n";
        return 2;
      }
      load_tensor = argv[++i];
    } else if (arg == "--load-all") {
      load_all = true;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      usage(argv[0]);
      return 2;
    }
  }

  try {
    cverl::HfModelLoader loader(model_dir);
    loader.load_metadata();
    const auto& cfg = loader.config();
    std::cout << "model_dir=" << loader.model_dir() << "\n";
    std::cout << "architecture=" << cfg.architecture << "\n";
    std::cout << "model_type=" << cfg.model_type << "\n";
    std::cout << "dtype=" << cfg.dtype << "\n";
    std::cout << "vocab_size=" << cfg.vocab_size << "\n";
    std::cout << "hidden_size=" << cfg.hidden_size << "\n";
    std::cout << "num_hidden_layers=" << cfg.num_hidden_layers << "\n";
    std::cout << "num_attention_heads=" << cfg.num_attention_heads << "\n";
    std::cout << "num_key_value_heads=" << cfg.num_key_value_heads << "\n";
    std::cout << "tensor_count=" << loader.tensors().size() << "\n";

    if (!find_substr.empty()) {
      size_t matches = 0;
      for (const auto& tensor : loader.tensors()) {
        if (tensor.name.find(find_substr) != std::string::npos) {
          std::cout << "match=" << tensor.name << " dtype=" << tensor.dtype << " shape=" << shape_string(tensor.shape)
                    << " file=" << tensor.filename << "\n";
          ++matches;
        }
      }
      std::cout << "match_count=" << matches << "\n";
    } else {
      size_t preview = std::min<size_t>(loader.tensors().size(), 12);
      for (size_t i = 0; i < preview; ++i) {
        const auto& tensor = loader.tensors()[i];
        std::cout << "tensor[" << i << "]=" << tensor.name << " dtype=" << tensor.dtype
                  << " shape=" << shape_string(tensor.shape) << " file=" << tensor.filename << "\n";
      }
    }

    if (!load_tensor.empty()) {
      torch::Tensor tensor = loader.load_tensor(load_tensor);
      std::cout << "loaded_tensor=" << load_tensor << "\n";
      std::cout << "loaded_dtype=" << tensor.dtype() << "\n";
      std::cout << "loaded_shape=" << tensor.sizes() << "\n";
      std::cout << "loaded_nbytes=" << tensor.nbytes() << "\n";
    }

    if (load_all) {
      auto tensors = loader.load_all_tensors();
      uint64_t total_bytes = 0;
      for (const auto& item : tensors) {
        total_bytes += static_cast<uint64_t>(item.second.nbytes());
      }
      std::cout << "loaded_all_tensor_count=" << tensors.size() << "\n";
      std::cout << "loaded_all_nbytes=" << total_bytes << "\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "inspect failed: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
