#include "cverl/torch/causal_lm_policy.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <torch/torch.h>

namespace {

void require(bool cond, const std::string& msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

void check_case(const std::string& model_dir,
                torch::ScalarType dtype,
                torch::Device device,
                const std::string& label) {
  cverl::torch_backend::CausalLmPolicyOptions opts;
  opts.kind = cverl::torch_backend::CausalLmPolicyOptions::Kind::kQwen3_5;
  opts.qwen_model_dir = model_dir;
  opts.qwen_max_layers = 0;
  opts.pad_id = 0;
  opts.param_dtype = dtype;
  auto policy = cverl::torch_backend::make_causal_lm_policy(opts);
  policy->to_device(device);

  auto params = policy->parameters();
  require(!params.empty(), label + ": no parameters");
  for (const auto& p : params) {
    require(p.scalar_type() == dtype, label + ": parameter dtype mismatch");
    require(p.device() == device, label + ": parameter device mismatch");
  }

  auto ids = torch::tensor({{1, 2}}, torch::TensorOptions().dtype(torch::kLong));
  auto response = torch::tensor({{3}}, torch::TensorOptions().dtype(torch::kLong));
  auto logits = policy->forward(ids, response);
  require(logits.scalar_type() == dtype, label + ": logits dtype mismatch");
  require(logits.device() == device, label + ": logits device mismatch");
}

}  // namespace

int main() {
  try {
    const char* env = std::getenv("CVERL_QWEN_MODEL_DIR");
    if (env == nullptr || std::string(env).empty()) {
      std::cout << "test_qwen3_5_param_dtype skipped: set CVERL_QWEN_MODEL_DIR\n";
      return 0;
    }
    const std::string model_dir = env;
    if (!std::filesystem::exists(std::filesystem::path(model_dir) / "config.json")) {
      std::cout << "test_qwen3_5_param_dtype skipped: missing config.json under "
                << model_dir << "\n";
      return 0;
    }

    check_case(model_dir, torch::kFloat32, torch::Device(torch::kCPU), "cpu-float32");
    if (torch::cuda::is_available()) {
      check_case(model_dir, torch::kBFloat16, torch::Device(torch::kCUDA, 0), "cuda-bfloat16");
      check_case(model_dir, torch::kFloat16, torch::Device(torch::kCUDA, 0), "cuda-float16");
    }
    std::cout << "test_qwen3_5_param_dtype passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_qwen3_5_param_dtype failed: " << e.what() << "\n";
    return 1;
  }
}
