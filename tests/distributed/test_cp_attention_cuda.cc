#include "cverl/distributed/context_parallel.h"
#include "cverl/distributed/cp_attention_cuda.h"

#include <torch/torch.h>

#include <iostream>
#include <stdexcept>
#include <string>

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
  auto cuda_opts = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat32);
  constexpr int64_t B = 1;
  constexpr int64_t H = 2;
  constexpr int64_t Tq = 3;
  constexpr int64_t Tk = 6;
  constexpr int64_t D = 8;
  constexpr int64_t V = 8;
  constexpr int64_t shard = 3;
  constexpr double scale = 0.35;

  auto q = torch::randn({B, H, Tq, D}, cuda_opts).contiguous().set_requires_grad(true);
  auto k = torch::randn({B, H, Tk, D}, cuda_opts).contiguous().set_requires_grad(true);
  auto v = torch::randn({B, H, Tk, V}, cuda_opts).contiguous().set_requires_grad(true);
  auto grad = torch::randn({B, H, Tq, V}, cuda_opts).contiguous();

  auto out = cverl::distributed::context_parallel_causal_attention_ring_blocks_recompute(
      q, k, v, {0, 3}, 0, Tk, shard, scale);

  auto q_ref = q.detach().clone().set_requires_grad(true);
  auto k_ref = k.detach().clone().set_requires_grad(true);
  auto v_ref = v.detach().clone().set_requires_grad(true);
  auto ref = cverl::distributed::context_parallel_causal_attention(q_ref, k_ref, v_ref, 0, scale);

  require_close(out.detach().cpu(), ref.detach().cpu(), 3.0e-4, 3.0e-4, "cp cuda forward");
  torch::autograd::backward({out}, {grad});
  torch::autograd::backward({ref}, {grad});
  require_close(q.grad().cpu(), q_ref.grad().cpu(), 5.0e-4, 5.0e-4, "cp cuda dq");
  require_close(k.grad().cpu(), k_ref.grad().cpu(), 5.0e-4, 5.0e-4, "cp cuda dk");
  require_close(v.grad().cpu(), v_ref.grad().cpu(), 5.0e-4, 5.0e-4, "cp cuda dv");

  std::cout << "test_cp_attention_cuda passed\n";
  return 0;
}
