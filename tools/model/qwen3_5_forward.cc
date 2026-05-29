#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "cverl/model/hf_model_loader.h"
#include "cverl/model/qwen3_5_text.h"

namespace {

struct Args {
  std::string model_dir;
  std::vector<int64_t> tokens{1, 2, 3, 4};
  int64_t layers = -1;
  bool logits = false;
  std::string save_output;
};

std::vector<int64_t> parse_tokens(const std::string& text) {
  std::vector<int64_t> out;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      out.push_back(std::stoll(item));
    }
  }
  if (out.empty()) {
    throw std::runtime_error("--tokens must contain at least one id");
  }
  return out;
}

Args parse_args(int argc, char** argv) {
  if (argc < 2) {
    throw std::runtime_error(
        "usage: qwen3_5_forward MODEL_DIR [--tokens 1,2,3] [--layers N] [--logits] [--save-output path.pt]");
  }
  Args args;
  args.model_dir = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string flag = argv[i];
    if (flag == "--tokens" && i + 1 < argc) {
      args.tokens = parse_tokens(argv[++i]);
    } else if (flag == "--layers" && i + 1 < argc) {
      args.layers = std::stoll(argv[++i]);
    } else if (flag == "--logits") {
      args.logits = true;
    } else if (flag == "--save-output" && i + 1 < argc) {
      args.save_output = argv[++i];
    } else {
      throw std::runtime_error("unknown or incomplete arg: " + flag);
    }
  }
  return args;
}

void print_shape(const torch::Tensor& tensor) {
  std::cout << "[";
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    if (i != 0) {
      std::cout << ", ";
    }
    std::cout << tensor.size(i);
  }
  std::cout << "]";
}

void save_tensor_binary(const torch::Tensor& tensor, const std::string& path) {
  auto out = tensor.detach().cpu().contiguous().to(torch::kFloat32);
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("failed to open output path: " + path);
  }
  const char magic[8] = {'C', 'V', 'T', 'E', 'N', 'S', 'R', '1'};
  file.write(magic, sizeof(magic));
  int64_t rank = out.dim();
  file.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
  for (int64_t i = 0; i < rank; ++i) {
    int64_t dim = out.size(i);
    file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
  }
  file.write(static_cast<const char*>(out.data_ptr()), static_cast<std::streamsize>(out.nbytes()));
  if (!file) {
    throw std::runtime_error("failed to write output tensor: " + path);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    torch::NoGradGuard no_grad;
    Args args = parse_args(argc, argv);
    cverl::HfModelLoader loader(args.model_dir);
    cverl::Qwen35TextModel model(std::move(loader));

    auto ids = torch::tensor(args.tokens, torch::kLong).view({1, static_cast<int64_t>(args.tokens.size())});
    auto out = args.logits ? model.forward_logits(ids, args.layers) : model.forward_hidden(ids, args.layers);
    if (!args.save_output.empty()) {
      save_tensor_binary(out, args.save_output);
    }
    std::cout << "model_dir=" << args.model_dir << "\n";
    std::cout << "tokens=" << args.tokens.size() << "\n";
    std::cout << "layers=" << (args.layers < 0 ? model.config().num_hidden_layers : args.layers) << "\n";
    std::cout << "output_shape=";
    print_shape(out);
    std::cout << "\n";
    if (!args.save_output.empty()) {
      std::cout << "saved_output=" << args.save_output << "\n";
    }
    if (args.logits) {
      auto last = out.index({0, static_cast<int64_t>(args.tokens.size()) - 1});
      auto top = std::get<1>(last.topk(5));
      std::cout << "top5_token_ids=" << top << "\n";
    } else {
      std::cout << "hidden_mean=" << out.mean().item<double>() << "\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
