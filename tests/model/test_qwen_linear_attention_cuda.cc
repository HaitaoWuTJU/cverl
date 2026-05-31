#include "cverl/model/qwen_linear_attention_cuda.h"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include <torch/torch.h>

namespace {

void require_close(const torch::Tensor& actual,
                   const torch::Tensor& expected,
                   double atol,
                   double rtol,
                   const char* name) {
  auto diff = (actual - expected).abs();
  double max_abs = diff.max().item<double>();
  double max_ref = expected.abs().max().item<double>();
  if (max_abs > atol + rtol * max_ref) {
    throw std::runtime_error(std::string(name) + " mismatch: max_abs=" + std::to_string(max_abs) +
                             ", max_ref=" + std::to_string(max_ref));
  }
}

}  // namespace

int main() {
  if (!torch::cuda::is_available()) {
    std::cout << "test_qwen_linear_attention_cuda skipped: CUDA is not available\n";
    return 0;
  }
  if (!cverl::qwen_linear_attention_cuda_available()) {
    std::cout << "test_qwen_linear_attention_cuda skipped: CUDA kernel was not built\n";
    return 0;
  }

  torch::manual_seed(7);
  auto opts = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat32);
  constexpr int B = 1;
  constexpr int H = 1;
  constexpr int T = 6;
  constexpr int K = 128;
  constexpr int V = 128;

  auto q = torch::randn({B, H, T, K}, opts).contiguous();
  auto k = torch::randn({B, H, T, K}, opts).contiguous();
  auto v = torch::randn({B, H, T, V}, opts).contiguous();
  auto beta = torch::sigmoid(torch::randn({B, H, T}, opts)).contiguous();
  auto g = (torch::randn({B, H, T}, opts) * 0.02).contiguous();
  auto grad_out = torch::randn({B, H, T, V}, opts).contiguous();

  auto saved_forward = cverl::qwen_linear_attention_cuda_forward(q, k, v, beta, g, true);
  auto recompute_forward = cverl::qwen_linear_attention_cuda_forward(q, k, v, beta, g, false);
  auto checkpointed_forward = cverl::qwen_linear_attention_cuda_forward_checkpointed(q, k, v, beta, g, 2);
  auto saved_out = std::get<0>(saved_forward);
  auto recompute_out = std::get<0>(recompute_forward);
  auto checkpointed_out = std::get<0>(checkpointed_forward);
  require_close(recompute_out.cpu(), saved_out.cpu(), 2e-4, 2e-4, "forward");
  require_close(checkpointed_out.cpu(), saved_out.cpu(), 2e-4, 2e-4, "checkpointed forward");

  auto saved_grads = cverl::qwen_linear_attention_cuda_backward(
      grad_out, q, k, v, beta, g, std::get<1>(saved_forward));
  auto recompute_grads = cverl::qwen_linear_attention_cuda_backward_recompute(
      grad_out, q, k, v, beta, g);
  setenv("CVERL_LINEAR_ATTN_CHUNK_REPLAY_BACKWARD", "0", 1);
  auto checkpointed_grads = cverl::qwen_linear_attention_cuda_backward_checkpointed(
      grad_out, q, k, v, beta, g, std::get<1>(checkpointed_forward), 2);
  unsetenv("CVERL_LINEAR_ATTN_CHUNK_REPLAY_BACKWARD");
  setenv("CVERL_LINEAR_ATTN_CHUNK_REPLAY_BACKWARD", "1", 1);
  auto chunk_replay_grads = cverl::qwen_linear_attention_cuda_backward_checkpointed(
      grad_out, q, k, v, beta, g, std::get<1>(checkpointed_forward), 2);
  unsetenv("CVERL_LINEAR_ATTN_CHUNK_REPLAY_BACKWARD");

  require_close(recompute_grads[0].cpu(), saved_grads[0].cpu(), 5e-4, 5e-4, "dq");
  require_close(recompute_grads[1].cpu(), saved_grads[1].cpu(), 5e-4, 5e-4, "dk");
  require_close(recompute_grads[2].cpu(), saved_grads[2].cpu(), 5e-4, 5e-4, "dv");
  require_close(recompute_grads[3].cpu(), saved_grads[3].cpu(), 5e-4, 5e-4, "dbeta");
  require_close(recompute_grads[4].cpu(), saved_grads[4].cpu(), 5e-4, 5e-4, "dg");
  require_close(checkpointed_grads[0].cpu(), saved_grads[0].cpu(), 5e-4, 5e-4, "checkpointed dq");
  require_close(checkpointed_grads[1].cpu(), saved_grads[1].cpu(), 5e-4, 5e-4, "checkpointed dk");
  require_close(checkpointed_grads[2].cpu(), saved_grads[2].cpu(), 5e-4, 5e-4, "checkpointed dv");
  require_close(checkpointed_grads[3].cpu(), saved_grads[3].cpu(), 5e-4, 5e-4, "checkpointed dbeta");
  require_close(checkpointed_grads[4].cpu(), saved_grads[4].cpu(), 5e-4, 5e-4, "checkpointed dg");
  require_close(chunk_replay_grads[0].cpu(), saved_grads[0].cpu(), 5e-4, 5e-4, "chunk replay dq");
  require_close(chunk_replay_grads[1].cpu(), saved_grads[1].cpu(), 5e-4, 5e-4, "chunk replay dk");
  require_close(chunk_replay_grads[2].cpu(), saved_grads[2].cpu(), 5e-4, 5e-4, "chunk replay dv");
  require_close(chunk_replay_grads[3].cpu(), saved_grads[3].cpu(), 5e-4, 5e-4, "chunk replay dbeta");
  require_close(chunk_replay_grads[4].cpu(), saved_grads[4].cpu(), 5e-4, 5e-4, "chunk replay dg");

  std::cout << "test_qwen_linear_attention_cuda passed\n";
  return 0;
}
