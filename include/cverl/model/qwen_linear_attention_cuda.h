#pragma once

#include <torch/torch.h>

namespace cverl {

bool qwen_linear_attention_cuda_available();

std::tuple<torch::Tensor, torch::Tensor> qwen_linear_attention_cuda_forward(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& beta,
    const torch::Tensor& g);

std::vector<torch::Tensor> qwen_linear_attention_cuda_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& beta,
    const torch::Tensor& g,
    const torch::Tensor& states);

}  // namespace cverl
