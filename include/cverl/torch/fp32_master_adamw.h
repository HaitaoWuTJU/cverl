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
};

class Fp32MasterAdamW {
 public:
  Fp32MasterAdamW(std::vector<torch::Tensor> model_parameters, Fp32MasterAdamWOptions options);

  void zero_grad();
  void step();

  const std::vector<torch::Tensor>& model_parameters() const { return model_parameters_; }
  const std::vector<torch::Tensor>& master_parameters() const { return master_parameters_; }

 private:
  std::vector<torch::Tensor> model_parameters_;
  std::vector<torch::Tensor> master_parameters_;
  std::vector<torch::Tensor> exp_avg_;
  std::vector<torch::Tensor> exp_avg_sq_;
  Fp32MasterAdamWOptions options_;
  int64_t step_ = 0;
};

std::vector<torch::Tensor> clone_detached(const std::vector<torch::Tensor>& parameters);

double parameter_delta_sum(const std::vector<torch::Tensor>& before, const std::vector<torch::Tensor>& after);

}  // namespace cverl::torch_backend
