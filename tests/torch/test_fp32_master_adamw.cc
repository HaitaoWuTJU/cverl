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

void require_allclose(const torch::Tensor& actual,
                      const torch::Tensor& expected,
                      const char* msg,
                      double rtol = 1.0e-5,
                      double atol = 1.0e-6) {
  if (!torch::allclose(actual, expected, rtol, atol)) {
    std::cerr << msg << "\nactual=" << actual << "\nexpected=" << expected << "\n";
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
    require_allclose(optimizer.grad_l2_norm_sq_tensor(),
                     torch::tensor({static_cast<float>(expected_tiny_grad_sq)}, torch::kFloat32),
                     "grad_l2_norm_sq_tensor before scaling");
    require_allclose(optimizer.grad_norm_sum_tensor(),
                     optimizer.main_grad_parameters()[0].norm().reshape({1}),
                     "grad_norm_sum_tensor before scaling");
    optimizer.scale_gradients(0.5);
    require(std::abs(optimizer.grad_l2_norm_sq() - expected_tiny_grad_sq * 0.25) < 1.0e-10,
            "grad_l2_norm_sq after scaling");
    optimizer.scale_gradients(torch::tensor({2.0f}, torch::kFloat32));
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

    cverl::torch_backend::Fp32MasterAdamWOptions flat_opts;
    flat_opts.lr = 3.0e-4;
    flat_opts.beta1 = 0.9;
    flat_opts.beta2 = 0.95;
    flat_opts.eps = 1.0e-8;
    flat_opts.weight_decay = 0.01;
    flat_opts.use_master_weights = true;
    auto dense_p = torch::linspace(-1.0, 1.0, 8, torch::TensorOptions().dtype(torch::kBFloat16));
    dense_p.set_requires_grad(true);
    auto flat_initial = dense_p.detach().to(torch::kFloat32).view({-1}).contiguous();
    auto flat_grad = torch::linspace(0.01, 0.08, 8, torch::TensorOptions().dtype(torch::kFloat32));
    cverl::torch_backend::Fp32MasterAdamW dense_optimizer({dense_p}, flat_opts);
    cverl::torch_backend::FlatAdamW flat_optimizer(flat_initial, flat_opts);
    dense_p.mutable_grad() = flat_grad.view_as(dense_p).to(dense_p.scalar_type());
    dense_optimizer.accumulate_model_grads();
    dense_optimizer.step();
    flat_optimizer.step(flat_grad);
    require_allclose(flat_optimizer.parameter_shard(), dense_optimizer.master_parameters()[0].view({-1}),
                     "flat AdamW must match dense master AdamW", 1.0e-5, 2.0e-5);
    require(flat_optimizer.exp_avg().scalar_type() == torch::kFloat32, "flat exp_avg must be fp32");
    require(flat_optimizer.exp_avg_sq().scalar_type() == torch::kFloat32, "flat exp_avg_sq must be fp32");
    require(flat_optimizer.step_count() == dense_optimizer.step_count(), "flat step count");

    cverl::torch_backend::FlatAdamW restored_flat(torch::zeros_like(flat_initial), flat_opts);
    restored_flat.load_state(flat_optimizer.parameter_shard(),
                             flat_optimizer.exp_avg(),
                             flat_optimizer.exp_avg_sq(),
                             flat_optimizer.step_count());
    require(restored_flat.step_count() == flat_optimizer.step_count(), "restored flat step count");
    require_allclose(restored_flat.parameter_shard(), flat_optimizer.parameter_shard(), "restored flat parameter");
    require_allclose(restored_flat.exp_avg(), flat_optimizer.exp_avg(), "restored flat exp_avg");
    require_allclose(restored_flat.exp_avg_sq(), flat_optimizer.exp_avg_sq(), "restored flat exp_avg_sq");

    cverl::torch_backend::Fp32MasterAdamWOptions flat_no_master_opts = flat_opts;
    flat_no_master_opts.use_master_weights = false;
    flat_no_master_opts.lr = 1.0e-2;
    auto dense_no_master_p =
        torch::linspace(-1.0, 1.0, 8, torch::TensorOptions().dtype(torch::kBFloat16));
    dense_no_master_p.set_requires_grad(true);
    auto flat_no_master_initial = dense_no_master_p.detach().view({-1}).contiguous();
    auto flat_no_master_grad = torch::linspace(0.1, 0.8, 8, torch::TensorOptions().dtype(torch::kFloat32));
    cverl::torch_backend::Fp32MasterAdamW dense_no_master_optimizer(
        {dense_no_master_p}, flat_no_master_opts);
    cverl::torch_backend::FlatAdamW flat_no_master_optimizer(
        flat_no_master_initial, flat_no_master_opts);
    dense_no_master_p.mutable_grad() =
        flat_no_master_grad.view_as(dense_no_master_p).to(dense_no_master_p.scalar_type());
    dense_no_master_optimizer.accumulate_model_grads();
    dense_no_master_optimizer.step();
    flat_no_master_optimizer.step(flat_no_master_grad);
    require(flat_no_master_optimizer.parameter_shard().scalar_type() == torch::kBFloat16,
            "flat no-master parameter shard keeps model dtype");
    require(flat_no_master_optimizer.exp_avg().scalar_type() == torch::kFloat32,
            "flat no-master exp_avg stays fp32");
    require(flat_no_master_optimizer.exp_avg_sq().scalar_type() == torch::kFloat32,
            "flat no-master exp_avg_sq stays fp32");
    require_allclose(flat_no_master_optimizer.parameter_shard().to(torch::kFloat32),
                     dense_no_master_p.detach().view({-1}).to(torch::kFloat32),
                     "flat no-master AdamW must match dense no-master AdamW",
                     1.0e-5,
                     2.0e-5);
    cverl::torch_backend::FlatAdamW restored_flat_no_master(
        torch::zeros_like(flat_no_master_initial), flat_no_master_opts);
    restored_flat_no_master.load_state(flat_no_master_optimizer.parameter_shard(),
                                       flat_no_master_optimizer.exp_avg(),
                                       flat_no_master_optimizer.exp_avg_sq(),
                                       flat_no_master_optimizer.step_count());
    require(restored_flat_no_master.parameter_shard().scalar_type() == torch::kBFloat16,
            "restored flat no-master parameter shard keeps model dtype");
    require(restored_flat_no_master.exp_avg().scalar_type() == torch::kFloat32,
            "restored flat no-master exp_avg stays fp32");
    require(restored_flat_no_master.exp_avg_sq().scalar_type() == torch::kFloat32,
            "restored flat no-master exp_avg_sq stays fp32");
    require(restored_flat_no_master.step_count() == flat_no_master_optimizer.step_count(),
            "restored flat no-master step count");
    require_allclose(restored_flat_no_master.parameter_shard().to(torch::kFloat32),
                     flat_no_master_optimizer.parameter_shard().to(torch::kFloat32),
                     "restored flat no-master parameter");

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
