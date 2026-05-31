#pragma once

#include <cstdint>
#include <vector>

#include <torch/torch.h>

namespace cverl::torch_backend {

struct Fp32MasterAdamWOptions {
  double lr = 1.0e-5;
  double beta1 = 0.9;
  double beta2 = 0.95;
  double eps = 1.0e-8;
  double weight_decay = 0.0;
  bool use_master_weights = true;
};

class Fp32MasterAdamW {
 public:
  Fp32MasterAdamW(std::vector<torch::Tensor> model_parameters, Fp32MasterAdamWOptions options);

  void zero_grad();
  void accumulate_model_grads(double scale = 1.0);
  void step();
  torch::Tensor grad_l2_norm_sq_tensor() const;
  double grad_l2_norm_sq() const;
  void scale_gradients(double scale);
  void load_state(const std::vector<torch::Tensor>& parameter_values,
                  const std::vector<torch::Tensor>& exp_avg,
                  const std::vector<torch::Tensor>& exp_avg_sq,
                  int64_t step);
  torch::Tensor grad_norm_sum_tensor() const;
  double grad_norm_sum() const;

  const std::vector<torch::Tensor>& model_parameters() const { return model_parameters_; }
  const std::vector<torch::Tensor>& master_parameters() const { return master_parameters_; }
  const std::vector<torch::Tensor>& main_grad_parameters() const { return main_grad_; }
  const std::vector<torch::Tensor>& exp_avg() const { return exp_avg_; }
  const std::vector<torch::Tensor>& exp_avg_sq() const { return exp_avg_sq_; }
  int64_t step_count() const { return step_; }
  const Fp32MasterAdamWOptions& options() const { return options_; }
  bool uses_master_weights() const { return options_.use_master_weights; }

 private:
  std::vector<torch::Tensor> model_parameters_;
  std::vector<torch::Tensor> master_parameters_;
  std::vector<torch::Tensor> main_grad_;
  std::vector<torch::Tensor> exp_avg_;
  std::vector<torch::Tensor> exp_avg_sq_;
  std::vector<bool> has_main_grad_;
  Fp32MasterAdamWOptions options_;
  int64_t step_ = 0;
};

class FlatAdamW {
 public:
  FlatAdamW(torch::Tensor parameter_shard, Fp32MasterAdamWOptions options);

  void step(const torch::Tensor& gradient_shard);
  void load_state(const torch::Tensor& parameter_value,
                  const torch::Tensor& exp_avg,
                  const torch::Tensor& exp_avg_sq,
                  int64_t step);

  const torch::Tensor& parameter_shard() const { return parameter_shard_; }
  const torch::Tensor& exp_avg() const { return exp_avg_; }
  const torch::Tensor& exp_avg_sq() const { return exp_avg_sq_; }
  int64_t step_count() const { return step_; }
  const Fp32MasterAdamWOptions& options() const { return options_; }

 private:
  torch::Tensor parameter_shard_;
  torch::Tensor exp_avg_;
  torch::Tensor exp_avg_sq_;
  Fp32MasterAdamWOptions options_;
  int64_t step_ = 0;
};

std::vector<torch::Tensor> clone_detached(const std::vector<torch::Tensor>& parameters);

double parameter_delta_sum(const std::vector<torch::Tensor>& before, const std::vector<torch::Tensor>& after);

}  // namespace cverl::torch_backend
