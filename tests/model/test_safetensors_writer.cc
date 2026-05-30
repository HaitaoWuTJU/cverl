#include "cverl/model/hf_model_loader.h"
#include "cverl/model/safetensors_writer.h"

#include <torch/torch.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

#include <unistd.h>

namespace {

void require(bool cond, const std::string& msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

void require_same_cpu_tensor(const torch::Tensor& actual,
                             const torch::Tensor& expected,
                             const std::string& label) {
  auto a = actual.detach().to(torch::kCPU).contiguous();
  auto e = expected.detach().to(torch::kCPU).contiguous();
  require(a.scalar_type() == e.scalar_type(), label + " dtype");
  require(a.sizes() == e.sizes(), label + " shape");
  require(a.nbytes() == e.nbytes(), label + " nbytes");
  require(std::memcmp(a.data_ptr(), e.data_ptr(), a.nbytes()) == 0, label + " bytes");
}

}  // namespace

int main() {
  try {
    namespace fs = std::filesystem;
    fs::path root = fs::temp_directory_path() / ("cverl_safetensors_writer_" + std::to_string(::getpid()));
    fs::create_directories(root);
    {
      std::ofstream cfg(root / "config.json");
      cfg << R"({"model_type":"unit","architectures":["UnitModel"],"dtype":"float32","vocab_size":8,"hidden_size":4,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1})";
    }

    auto a = torch::arange(6, torch::TensorOptions().dtype(torch::kFloat32)).reshape({2, 3});
    auto b = torch::tensor({1.0f, -2.0f, 3.5f}, torch::TensorOptions().dtype(torch::kFloat32));
    std::unordered_map<std::string, torch::Tensor> tensors;
    tensors.emplace("model.a.weight", a);
    tensors.emplace("model.b.bias", b);
    cverl::write_safetensors((root / "model.safetensors").string(), tensors, "float32");

    cverl::HfModelLoader loader(root.string());
    loader.load_metadata();
    require(loader.tensors().size() == 2, "tensor count");
    require_same_cpu_tensor(loader.load_tensor("model.a.weight"), a, "tensor a round trip");
    require_same_cpu_tensor(loader.load_tensor("model.b.bias"), b, "tensor b round trip");

    std::cout << "safetensors writer test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_safetensors_writer failed: " << e.what() << "\n";
    return 1;
  }
}
