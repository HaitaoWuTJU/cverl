#include "cverl/model/qwen_linear_attention_cuda.h"

#include <stdexcept>

namespace cverl {

#ifndef CVERL_ENABLE_CUDA
bool qwen_linear_attention_cuda_available() {
  return false;
}

std::tuple<torch::Tensor, torch::Tensor> qwen_linear_attention_cuda_forward(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& beta,
    const torch::Tensor& g) {
  (void)query;
  (void)key;
  (void)value;
  (void)beta;
  (void)g;
  throw std::runtime_error("Qwen linear attention CUDA kernel was not built");
}

std::vector<torch::Tensor> qwen_linear_attention_cuda_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& beta,
    const torch::Tensor& g,
    const torch::Tensor& states) {
  (void)grad_out;
  (void)query;
  (void)key;
  (void)value;
  (void)beta;
  (void)g;
  (void)states;
  throw std::runtime_error("Qwen linear attention CUDA kernel was not built");
}
#endif

}  // namespace cverl
