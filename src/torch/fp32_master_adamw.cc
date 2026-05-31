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
  main_grad_.reserve(model_parameters_.size());
  exp_avg_.reserve(model_parameters_.size());
  exp_avg_sq_.reserve(model_parameters_.size());
  has_main_grad_.reserve(model_parameters_.size());
  torch::NoGradGuard no_grad;
  for (const auto& p : model_parameters_) {
    if (!p.defined()) {
      throw std::invalid_argument("Fp32MasterAdamW: undefined parameter");
    }
    auto master = p.detach().to(torch::kFloat32).contiguous();
    if (options_.use_master_weights) {
      master_parameters_.push_back(master);
    }
    main_grad_.push_back(torch::zeros_like(master));
    exp_avg_.push_back(torch::zeros_like(master));
    exp_avg_sq_.push_back(torch::zeros_like(master));
    has_main_grad_.push_back(false);
  }
}

void Fp32MasterAdamW::zero_grad() {
  torch::NoGradGuard no_grad;
  for (size_t i = 0; i < model_parameters_.size(); ++i) {
    auto& p = model_parameters_[i];
    if (p.grad().defined()) {
      p.mutable_grad().zero_();
    }
    main_grad_[i].zero_();
    has_main_grad_[i] = false;
  }
}

void Fp32MasterAdamW::accumulate_model_grads(double scale) {
  torch::NoGradGuard no_grad;
  for (size_t i = 0; i < model_parameters_.size(); ++i) {
    auto& model = model_parameters_[i];
    if (!model.grad().defined()) {
      continue;
    }
    main_grad_[i].add_(model.grad().detach().to(torch::kFloat32), scale);
    model.mutable_grad().zero_();
    has_main_grad_[i] = true;
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
    torch::Tensor grad;
    if (has_main_grad_[i]) {
      grad = main_grad_[i];
    } else if (model.grad().defined()) {
      grad = model.grad().detach().to(torch::kFloat32);
    } else {
      continue;
    }
    auto master = options_.use_master_weights ? master_parameters_[i] : model.detach().to(torch::kFloat32).contiguous();
    auto& m = exp_avg_[i];
    auto& v = exp_avg_sq_[i];

    m.mul_(beta1).add_(grad, 1.0 - beta1);
    v.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);

    if (options_.weight_decay != 0.0) {
      master.add_(master, -options_.lr * options_.weight_decay);
    }
    auto denom = (v / bias_correction2).sqrt_().add_(options_.eps);
    master.addcdiv_(m, denom, -step_size);
    if (options_.use_master_weights) {
      master_parameters_[i].copy_(master);
    }
    model.copy_(master.to(model.scalar_type()));
  }
}

void Fp32MasterAdamW::load_state(const std::vector<torch::Tensor>& parameter_values,
                                 const std::vector<torch::Tensor>& exp_avg,
                                 const std::vector<torch::Tensor>& exp_avg_sq,
                                 int64_t step) {
  if (step < 0) {
    throw std::invalid_argument("Fp32MasterAdamW::load_state: negative step");
  }
  if (parameter_values.size() != model_parameters_.size() ||
      exp_avg.size() != model_parameters_.size() ||
      exp_avg_sq.size() != model_parameters_.size()) {
    throw std::invalid_argument("Fp32MasterAdamW::load_state: state size mismatch");
  }
  torch::NoGradGuard no_grad;
  for (size_t i = 0; i < model_parameters_.size(); ++i) {
    auto& model = model_parameters_[i];
    const auto& value = parameter_values[i];
    if (!value.defined() || value.sizes() != model.sizes()) {
      throw std::invalid_argument("Fp32MasterAdamW::load_state: parameter shape mismatch");
    }
    if (!exp_avg[i].defined() || exp_avg[i].sizes() != main_grad_[i].sizes() ||
        !exp_avg_sq[i].defined() || exp_avg_sq[i].sizes() != main_grad_[i].sizes()) {
      throw std::invalid_argument("Fp32MasterAdamW::load_state: optimizer state shape mismatch");
    }

    if (options_.use_master_weights) {
      master_parameters_[i].copy_(value.detach().to(master_parameters_[i].device(), torch::kFloat32));
      model.copy_(master_parameters_[i].to(model.scalar_type()));
    } else {
      model.copy_(value.detach().to(model.device(), model.scalar_type()));
    }
    exp_avg_[i].copy_(exp_avg[i].detach().to(exp_avg_[i].device(), torch::kFloat32));
    exp_avg_sq_[i].copy_(exp_avg_sq[i].detach().to(exp_avg_sq_[i].device(), torch::kFloat32));
    main_grad_[i].zero_();
    has_main_grad_[i] = false;
    if (model.grad().defined()) {
      model.mutable_grad().zero_();
    }
  }
  step_ = step;
}

double Fp32MasterAdamW::grad_norm_sum() const {
  double out = 0.0;
  for (size_t i = 0; i < model_parameters_.size(); ++i) {
    if (has_main_grad_[i]) {
      out += main_grad_[i].detach().norm().item<double>();
    } else if (model_parameters_[i].grad().defined()) {
      out += model_parameters_[i].grad().detach().to(torch::kFloat32).norm().item<double>();
    }
  }
  return out;
}

}  // namespace cverl::torch_backend
