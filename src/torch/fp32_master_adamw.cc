#include "cverl/torch/fp32_master_adamw.h"

#include <cmath>
#include <stdexcept>

namespace cverl::torch_backend {
namespace {

void validate_adamw_options(const Fp32MasterAdamWOptions& options, const char* name) {
  if (options.lr < 0.0 || options.beta1 < 0.0 || options.beta1 >= 1.0 ||
      options.beta2 < 0.0 || options.beta2 >= 1.0 || options.eps <= 0.0 ||
      options.weight_decay < 0.0) {
    throw std::invalid_argument(std::string(name) + ": invalid optimizer options");
  }
}

void add_scalar_term(torch::Tensor* total, const torch::Tensor& term) {
  if (!term.defined()) {
    return;
  }
  auto scalar = term.detach().to(torch::kFloat32);
  if (!total->defined()) {
    *total = scalar;
  } else if (total->device() == scalar.device()) {
    *total = *total + scalar;
  } else {
    *total = *total + scalar.to(total->device());
  }
}

double scalar_total_to_double(const torch::Tensor& total) {
  if (!total.defined()) {
    return 0.0;
  }
  return total.item<double>();
}

torch::Tensor scalar_total_or_zero(const torch::Tensor& total, torch::Device fallback_device) {
  if (total.defined()) {
    return total.to(torch::kFloat32).reshape({1});
  }
  return torch::zeros({1}, torch::TensorOptions().device(fallback_device).dtype(torch::kFloat32));
}

}  // namespace

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
  torch::Tensor total;
  for (size_t i = 0; i < before.size(); ++i) {
    auto delta = (after[i].detach().to(torch::kFloat32) - before[i].to(torch::kFloat32)).abs().sum();
    add_scalar_term(&total, delta);
  }
  return scalar_total_to_double(total);
}

Fp32MasterAdamW::Fp32MasterAdamW(std::vector<torch::Tensor> model_parameters,
                                 Fp32MasterAdamWOptions options)
    : model_parameters_(std::move(model_parameters)), options_(options) {
  validate_adamw_options(options_, "Fp32MasterAdamW");
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

torch::Tensor Fp32MasterAdamW::grad_l2_norm_sq_tensor() const {
  torch::Tensor total;
  torch::Device fallback_device(torch::kCPU);
  for (size_t i = 0; i < model_parameters_.size(); ++i) {
    if (model_parameters_[i].defined()) {
      fallback_device = model_parameters_[i].device();
    }
    torch::Tensor grad;
    if (has_main_grad_[i]) {
      grad = main_grad_[i];
    } else if (model_parameters_[i].grad().defined()) {
      grad = model_parameters_[i].grad().detach().to(torch::kFloat32);
    } else {
      continue;
    }
    add_scalar_term(&total, grad.pow(2).sum());
  }
  return scalar_total_or_zero(total, fallback_device);
}

double Fp32MasterAdamW::grad_l2_norm_sq() const {
  return scalar_total_to_double(grad_l2_norm_sq_tensor());
}

void Fp32MasterAdamW::scale_gradients(double scale) {
  if (!std::isfinite(scale) || scale < 0.0) {
    throw std::invalid_argument("Fp32MasterAdamW::scale_gradients: invalid scale");
  }
  torch::NoGradGuard no_grad;
  for (size_t i = 0; i < model_parameters_.size(); ++i) {
    if (has_main_grad_[i]) {
      main_grad_[i].mul_(scale);
    } else if (model_parameters_[i].grad().defined()) {
      model_parameters_[i].mutable_grad().mul_(scale);
    }
  }
}

void Fp32MasterAdamW::scale_gradients(const torch::Tensor& scale) {
  if (!scale.defined() || scale.numel() != 1) {
    throw std::invalid_argument("Fp32MasterAdamW::scale_gradients: tensor scale must be a defined scalar");
  }
  torch::NoGradGuard no_grad;
  for (size_t i = 0; i < model_parameters_.size(); ++i) {
    if (has_main_grad_[i]) {
      main_grad_[i].mul_(scale.to(main_grad_[i].device(), main_grad_[i].scalar_type()));
    } else if (model_parameters_[i].grad().defined()) {
      auto& grad = model_parameters_[i].mutable_grad();
      grad.mul_(scale.to(grad.device(), grad.scalar_type()));
    }
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

torch::Tensor Fp32MasterAdamW::grad_norm_sum_tensor() const {
  torch::Tensor total;
  torch::Device fallback_device(torch::kCPU);
  for (size_t i = 0; i < model_parameters_.size(); ++i) {
    if (model_parameters_[i].defined()) {
      fallback_device = model_parameters_[i].device();
    }
    if (has_main_grad_[i]) {
      add_scalar_term(&total, main_grad_[i].detach().norm());
    } else if (model_parameters_[i].grad().defined()) {
      add_scalar_term(&total, model_parameters_[i].grad().detach().to(torch::kFloat32).norm());
    }
  }
  return scalar_total_or_zero(total, fallback_device);
}

double Fp32MasterAdamW::grad_norm_sum() const {
  return scalar_total_to_double(grad_norm_sum_tensor());
}

FlatAdamW::FlatAdamW(torch::Tensor parameter_shard, Fp32MasterAdamWOptions options)
    : options_(options) {
  validate_adamw_options(options_, "FlatAdamW");
  if (!parameter_shard.defined() || !parameter_shard.is_contiguous()) {
    throw std::invalid_argument("FlatAdamW: parameter shard must be contiguous");
  }
  torch::NoGradGuard no_grad;
  parameter_shard_ = parameter_shard.detach().to(torch::kFloat32).contiguous();
  exp_avg_ = torch::zeros_like(parameter_shard_);
  exp_avg_sq_ = torch::zeros_like(parameter_shard_);
}

void FlatAdamW::step(const torch::Tensor& gradient_shard) {
  if (!gradient_shard.defined() || gradient_shard.numel() != parameter_shard_.numel()) {
    throw std::invalid_argument("FlatAdamW::step: gradient shard shape mismatch");
  }
  torch::NoGradGuard no_grad;
  ++step_;
  auto grad = gradient_shard.detach().to(parameter_shard_.device(), torch::kFloat32).view(parameter_shard_.sizes());
  const double beta1 = options_.beta1;
  const double beta2 = options_.beta2;
  const double bias_correction1 = 1.0 - std::pow(beta1, static_cast<double>(step_));
  const double bias_correction2 = 1.0 - std::pow(beta2, static_cast<double>(step_));
  const double step_size = options_.lr / bias_correction1;

  exp_avg_.mul_(beta1).add_(grad, 1.0 - beta1);
  exp_avg_sq_.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);
  if (options_.weight_decay != 0.0) {
    parameter_shard_.add_(parameter_shard_, -options_.lr * options_.weight_decay);
  }
  auto denom = (exp_avg_sq_ / bias_correction2).sqrt_().add_(options_.eps);
  parameter_shard_.addcdiv_(exp_avg_, denom, -step_size);
}

void FlatAdamW::load_state(const torch::Tensor& parameter_value,
                           const torch::Tensor& exp_avg,
                           const torch::Tensor& exp_avg_sq,
                           int64_t step) {
  if (step < 0) {
    throw std::invalid_argument("FlatAdamW::load_state: negative step");
  }
  if (!parameter_value.defined() || parameter_value.numel() != parameter_shard_.numel() ||
      !exp_avg.defined() || exp_avg.numel() != parameter_shard_.numel() ||
      !exp_avg_sq.defined() || exp_avg_sq.numel() != parameter_shard_.numel()) {
    throw std::invalid_argument("FlatAdamW::load_state: state shape mismatch");
  }
  torch::NoGradGuard no_grad;
  parameter_shard_.copy_(parameter_value.detach().to(parameter_shard_.device(), torch::kFloat32).view(parameter_shard_.sizes()));
  exp_avg_.copy_(exp_avg.detach().to(exp_avg_.device(), torch::kFloat32).view(exp_avg_.sizes()));
  exp_avg_sq_.copy_(exp_avg_sq.detach().to(exp_avg_sq_.device(), torch::kFloat32).view(exp_avg_sq_.sizes()));
  step_ = step;
}

}  // namespace cverl::torch_backend
