#pragma once

#include <string>
#include <unordered_map>

#include <torch/torch.h>

namespace cverl {

// Write a single-file safetensors checkpoint. Tensor keys are HF parameter
// names and values may live on CPU or CUDA; tensors are copied to CPU and made
// contiguous before writing.
void write_safetensors(
    const std::string& path,
    const std::unordered_map<std::string, torch::Tensor>& tensors,
    const std::string& dtype = "float32");

}  // namespace cverl
