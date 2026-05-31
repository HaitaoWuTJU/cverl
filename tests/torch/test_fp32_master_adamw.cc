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
    const double expected_tiny_grad_sq =
        optimizer.main_grad_parameters()[0].pow(2).sum().item<double>();
    const double tiny_grad_sq = optimizer.grad_l2_norm_sq();
    require(std::abs(tiny_grad_sq - expected_tiny_grad_sq) < 1.0e-10, "grad_l2_norm_sq before scaling");
    optimizer.scale_gradients(0.5);
    require(std::abs(optimizer.grad_l2_norm_sq() - expected_tiny_grad_sq * 0.25) < 1.0e-10,
            "grad_l2_norm_sq after scaling");
    optimizer.scale_gradients(2.0);
    require(optimizer.main_grad_parameters()[0].scalar_type() == torch::kFloat32, "main_grad must be fp32");
    require(p.grad().defined(), "accumulate_model_grads keeps grad tensor allocated");
    require(p.grad().to(torch::kFloat32).abs().sum().item<double>() == 0.0, "accumulate_model_grads clears model gradient");
    optimizer.step();
    require(optimizer.step_count() == 1, "optimizer step count");
    require(optimizer.exp_avg().size() == 1 && optimizer.exp_avg_sq().size() == 1, "optimizer state tensors exposed");
    require(optimizer.exp_avg()[0].scalar_type() == torch::kFloat32, "exp_avg must be fp32");
    require(optimizer.exp_avg_sq()[0].scalar_type() == torch::kFloat32, "exp_avg_sq must be fp32");
    require(optimizer.options().lr == opts.lr, "optimizer options exposed");

    auto restored_p = torch::empty_like(p);
    restored_p.set_requires_grad(true);
    cverl::torch_backend::Fp32MasterAdamW restored_optimizer({restored_p}, opts);
    restored_optimizer.load_state(
        cverl::torch_backend::clone_detached(optimizer.master_parameters()),
        cverl::torch_backend::clone_detached(optimizer.exp_avg()),
        cverl::torch_backend::clone_detached(optimizer.exp_avg_sq()),
        optimizer.step_count());
    require(restored_optimizer.step_count() == optimizer.step_count(), "restored optimizer step count");
    require(torch::allclose(restored_optimizer.master_parameters()[0], optimizer.master_parameters()[0]),
            "restored master parameter");
    require(torch::allclose(restored_optimizer.exp_avg()[0], optimizer.exp_avg()[0]), "restored exp_avg");
    require(torch::allclose(restored_optimizer.exp_avg_sq()[0], optimizer.exp_avg_sq()[0]), "restored exp_avg_sq");

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

    auto q = torch::ones({4}, torch::TensorOptions().dtype(torch::kBFloat16));
    q.set_requires_grad(true);
    opts.use_master_weights = false;
    opts.lr = 1.0e-2;
    cverl::torch_backend::Fp32MasterAdamW no_master_optimizer({q}, opts);
    require(no_master_optimizer.master_parameters().empty(), "no-master mode must not allocate fp32 master weights");
    q.mutable_grad() = torch::full_like(q, 1.0e-1);
    no_master_optimizer.accumulate_model_grads();
    auto q_before = q.detach().clone();
    no_master_optimizer.step();
    const double no_master_delta = (q.detach().to(torch::kFloat32) - q_before.to(torch::kFloat32)).abs().sum().item<double>();
    require(no_master_delta > 0.0, "no-master BF16 mode must update model parameter when update exceeds BF16 quantum");
    auto restored_q = torch::empty_like(q);
    restored_q.set_requires_grad(true);
    cverl::torch_backend::Fp32MasterAdamW restored_no_master({restored_q}, opts);
    std::vector<torch::Tensor> no_master_params = {q};
    restored_no_master.load_state(
        cverl::torch_backend::clone_detached(no_master_params),
        cverl::torch_backend::clone_detached(no_master_optimizer.exp_avg()),
        cverl::torch_backend::clone_detached(no_master_optimizer.exp_avg_sq()),
        no_master_optimizer.step_count());
    require(restored_no_master.master_parameters().empty(), "restored no-master mode keeps master weights disabled");
    require(torch::allclose(restored_q.to(torch::kFloat32), q.to(torch::kFloat32)), "restored no-master parameter");

    std::cout << "fp32 master AdamW test passed"
              << " master_delta=" << master_delta
              << " model_delta=" << model_delta
              << " no_master_delta=" << no_master_delta << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_fp32_master_adamw failed: " << e.what() << "\n";
    return 1;
  }
}
