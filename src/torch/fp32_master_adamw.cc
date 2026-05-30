#include "cverl/torch/fp32_master_adamw.h"

#include <cmath>
#include <stdexcept>

namespace cverl::torch_backend {

std::vector<torch::Tensor> clone_detached(const std::vector<torch::Tensor>& parameters) {
  std::vector<torch::Tensor> out;
  out.reserve(parameters.size());
  for (const auto& p : parameters) {
    out.push_back(p.detach().clone());
  }
  return out;
}

double parameter_delta_sum(const std::vector<torch::Tensor>& before, const std::vector<torch::Tensor>& after) {
  if (before.size() != after.size()) {
    throw std::invalid_argument("parameter_delta_sum: size mismatch");
  }
  double out = 0.0;
  for (size_t i = 0; i < before.size(); ++i) {
    out += (after[i].detach().to(torch::kFloat32) - before[i].to(torch::kFloat32)).abs().sum().item<double>();
  }
  return out;
}

Fp32MasterAdamW::Fp32MasterAdamW(std::vector<torch::Tensor> model_parameters,
                                 Fp32MasterAdamWOptions options)
    : model_parameters_(std::move(model_parameters)), options_(options) {
  if (options_.lr < 0.0 || options_.beta1 < 0.0 || options_.beta1 >= 1.0 ||
      options_.beta2 < 0.0 || options_.beta2 >= 1.0 || options_.eps <= 0.0 ||
      options_.weight_decay < 0.0) {
    throw std::invalid_argument("Fp32MasterAdamW: invalid optimizer options");
  }
  master_parameters_.reserve(model_parameters_.size());
  exp_avg_.reserve(model_parameters_.size());
  exp_avg_sq_.reserve(model_parameters_.size());
  torch::NoGradGuard no_grad;
  for (const auto& p : model_parameters_) {
    if (!p.defined()) {
      throw std::invalid_argument("Fp32MasterAdamW: undefined parameter");
    }
    auto master = p.detach().to(torch::kFloat32).contiguous();
    master_parameters_.push_back(master);
    exp_avg_.push_back(torch::zeros_like(master));
    exp_avg_sq_.push_back(torch::zeros_like(master));
  }
}

void Fp32MasterAdamW::zero_grad() {
  torch::NoGradGuard no_grad;
  for (auto& p : model_parameters_) {
    if (p.grad().defined()) {
      p.mutable_grad().zero_();
    }
  }
}

void Fp32MasterAdamW::step() {
  torch::NoGradGuard no_grad;
  ++step_;
  const double beta1 = options_.beta1;
  const double beta2 = options_.beta2;
  const double bias_correction1 = 1.0 - std::pow(beta1, static_cast<double>(step_));
  const double bias_correction2 = 1.0 - std::pow(beta2, static_cast<double>(step_));
  const double step_size = options_.lr / bias_correction1;

  for (size_t i = 0; i < model_parameters_.size(); ++i) {
    auto& model = model_parameters_[i];
    if (!model.grad().defined()) {
      continue;
    }
    auto grad = model.grad().detach().to(torch::kFloat32);
    auto& master = master_parameters_[i];
    auto& m = exp_avg_[i];
    auto& v = exp_avg_sq_[i];

    m.mul_(beta1).add_(grad, 1.0 - beta1);
    v.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);

    if (options_.weight_decay != 0.0) {
      master.add_(master, -options_.lr * options_.weight_decay);
    }
    auto denom = (v / bias_correction2).sqrt_().add_(options_.eps);
    master.addcdiv_(m, denom, -step_size);
    model.copy_(master.to(model.scalar_type()));
  }
}

}  // namespace cverl::torch_backend
