#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <torch/torch.h>

#include "cverl/torch/fp32_master_adamw.h"

namespace {

void require(bool cond, const char* msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

}  // namespace

int main() {
  try {
    auto p = torch::ones({4}, torch::TensorOptions().dtype(torch::kBFloat16));
    p.set_requires_grad(true);

    cverl::torch_backend::Fp32MasterAdamWOptions opts;
    opts.lr = 1.0e-4;
    opts.beta1 = 0.9;
    opts.beta2 = 0.95;
    opts.eps = 1.0e-8;
    opts.weight_decay = 0.0;
    cverl::torch_backend::Fp32MasterAdamW optimizer({p}, opts);

    auto model_before = p.detach().clone();
    auto master_before = cverl::torch_backend::clone_detached(optimizer.master_parameters());
    p.mutable_grad() = torch::full_like(p, 1.0e-3);
    optimizer.accumulate_model_grads();
    require(optimizer.main_grad_parameters()[0].scalar_type() == torch::kFloat32, "main_grad must be fp32");
    require(p.grad().defined(), "accumulate_model_grads keeps grad tensor allocated");
    require(p.grad().to(torch::kFloat32).abs().sum().item<double>() == 0.0, "accumulate_model_grads clears model gradient");
    optimizer.step();

    const double model_delta =
        (p.detach().to(torch::kFloat32) - model_before.to(torch::kFloat32)).abs().sum().item<double>();
    const double master_delta =
        cverl::torch_backend::parameter_delta_sum(master_before, optimizer.master_parameters());

    require(master_delta > 0.0, "fp32 master parameter must move");
    require(model_delta == 0.0, "small update should be preserved in fp32 master even if bf16 model rounds it away");
    require(optimizer.master_parameters()[0].scalar_type() == torch::kFloat32, "master parameter must be fp32");

    optimizer.zero_grad();
    require(p.grad().defined(), "zero_grad keeps grad tensor allocated");
    require(p.grad().to(torch::kFloat32).abs().sum().item<double>() == 0.0, "zero_grad clears model gradient");

    std::cout << "fp32 master AdamW test passed"
              << " master_delta=" << master_delta
              << " model_delta=" << model_delta << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_fp32_master_adamw failed: " << e.what() << "\n";
    return 1;
  }
}
