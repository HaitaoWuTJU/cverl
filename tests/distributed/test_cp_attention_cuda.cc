#include "cverl/distributed/context_parallel.h"
#include "cverl/distributed/cp_attention_cuda.h"

#include <torch/torch.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_close(const torch::Tensor& actual,
                   const torch::Tensor& expected,
                   double atol,
                   double rtol,
                   const std::string& name) {
  auto diff = (actual - expected).abs();
  const double max_abs = diff.max().item<double>();
  const double max_ref = expected.abs().max().item<double>();
  if (max_abs > atol + rtol * max_ref) {
    throw std::runtime_error(name + " mismatch: max_abs=" + std::to_string(max_abs) +
                             " max_ref=" + std::to_string(max_ref));
  }
}

void run_case(const torch::Tensor& q_src,
              const torch::Tensor& k_src,
              const torch::Tensor& v_src,
              const torch::Tensor& grad_src,
              const std::vector<int64_t>& key_begin_positions,
              int64_t query_begin,
              int64_t original_sequence_length,
              int64_t shard,
              double scale,
              double atol,
              double rtol,
              const std::string& name) {
  auto q = q_src.detach().clone().set_requires_grad(true);
  auto k = k_src.detach().clone().set_requires_grad(true);
  auto v = v_src.detach().clone().set_requires_grad(true);
  auto grad = grad_src.detach().clone();
  auto out = cverl::distributed::context_parallel_causal_attention_ring_blocks_recompute(
      q, k, v, key_begin_positions, query_begin, original_sequence_length, shard, scale);

  auto q_ref = q_src.detach().clone().set_requires_grad(true);
  std::vector<torch::Tensor> k_ref_blocks;
  std::vector<torch::Tensor> v_ref_blocks;
  k_ref_blocks.reserve(key_begin_positions.size());
  v_ref_blocks.reserve(key_begin_positions.size());
  for (int64_t i = 0; i < static_cast<int64_t>(key_begin_positions.size()); ++i) {
    k_ref_blocks.push_back(k_src.detach().clone().narrow(2, i * shard, shard).contiguous().set_requires_grad(true));
    v_ref_blocks.push_back(v_src.detach().clone().narrow(2, i * shard, shard).contiguous().set_requires_grad(true));
  }
  auto ref = cverl::distributed::context_parallel_causal_attention_streaming_blocks(
      q_ref, k_ref_blocks, v_ref_blocks, key_begin_positions, query_begin, original_sequence_length, scale);

  require_close(out.detach().to(torch::kFloat32).cpu(), ref.detach().to(torch::kFloat32).cpu(), atol, rtol, name + " forward");
  const int64_t valid_queries =
      std::max<int64_t>(0, std::min<int64_t>(q_src.size(2), original_sequence_length - query_begin));
  if (valid_queries < q_src.size(2)) {
    const auto padded = out.detach().narrow(2, valid_queries, q_src.size(2) - valid_queries);
    require_close(padded.to(torch::kFloat32).cpu(),
                  torch::zeros_like(padded).to(torch::kFloat32).cpu(),
                  0.0,
                  0.0,
                  name + " padded query output");
  }
  torch::autograd::backward({out}, {grad});
  torch::autograd::backward({ref}, {grad});
  require_close(q.grad().to(torch::kFloat32).cpu(), q_ref.grad().to(torch::kFloat32).cpu(), atol, rtol, name + " dq");
  for (int64_t i = 0; i < static_cast<int64_t>(key_begin_positions.size()); ++i) {
    require_close(k.grad().narrow(2, i * shard, shard).to(torch::kFloat32).cpu(),
                  k_ref_blocks.at(static_cast<size_t>(i)).grad().to(torch::kFloat32).cpu(),
                  atol,
                  rtol,
                  name + " dk block " + std::to_string(i));
    require_close(v.grad().narrow(2, i * shard, shard).to(torch::kFloat32).cpu(),
                  v_ref_blocks.at(static_cast<size_t>(i)).grad().to(torch::kFloat32).cpu(),
                  atol,
                  rtol,
                  name + " dv block " + std::to_string(i));
  }
}

void run_dtype_cases(torch::ScalarType dtype,
                     double atol,
                     double rtol,
                     const std::string& dtype_name) {
  auto opts = torch::TensorOptions().device(torch::kCUDA, 0).dtype(dtype);
  constexpr int64_t B = 1;
  constexpr int64_t H = 2;
  constexpr int64_t Tq = 3;
  constexpr int64_t Tk = 6;
  constexpr int64_t D = 8;
  constexpr int64_t V = 8;
  constexpr int64_t shard = 3;
  constexpr double scale = 0.35;

  auto q = torch::randn({B, H, Tq, D}, opts).contiguous().set_requires_grad(true);
  auto k = torch::randn({B, H, Tk, D}, opts).contiguous().set_requires_grad(true);
  auto v = torch::randn({B, H, Tk, V}, opts).contiguous().set_requires_grad(true);
  auto grad = torch::randn({B, H, Tq, V}, opts).contiguous();

  run_case(q, k, v, grad, {0, 3}, 0, Tk, shard, scale, atol, rtol, dtype_name + " dense-order");
  run_case(q, k, v, grad, {3, 0}, 0, Tk, shard, scale, atol, rtol, dtype_name + " ring-order");
  run_case(q, k, v, grad, {3, 0}, 3, 5, shard, scale, atol, rtol, dtype_name + " padded-tail");
}

}  // namespace

int main() {
  if (!torch::cuda::is_available()) {
    std::cout << "test_cp_attention_cuda skipped: CUDA is not available\n";
    return 0;
  }
  if (!cverl::distributed::cp_attention_cuda_available()) {
    std::cout << "test_cp_attention_cuda skipped: CUDA kernel was not built\n";
    return 0;
  }

  torch::manual_seed(31);
  run_dtype_cases(torch::kFloat32, 5.0e-4, 5.0e-4, "cp cuda fp32");
  run_dtype_cases(torch::kFloat16, 3.0e-3, 3.0e-3, "cp cuda fp16");
  run_dtype_cases(torch::kBFloat16, 2.0e-2, 2.0e-2, "cp cuda bf16");

  std::cout << "test_cp_attention_cuda passed\n";
  return 0;
}
