#include "cverl/cverl.h"
#include "cverl/torch/core_algos_torch.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

cverl_const_tensor2d_t ct(const std::vector<float>& v, int64_t rows, int64_t cols) {
  return cverl_const_tensor2d_t{v.data(), CVERL_DTYPE_F32, CVERL_DEVICE_CPU, rows, cols};
}

cverl_tensor2d_t mt(std::vector<float>& v, int64_t rows, int64_t cols) {
  return cverl_tensor2d_t{v.data(), CVERL_DTYPE_F32, CVERL_DEVICE_CPU, rows, cols};
}

void require_close(float actual, float expected, float atol, const char* context) {
  if (std::fabs(actual - expected) > atol) {
    std::cerr << context << " expected " << expected << " got " << actual << "\n";
    std::exit(1);
  }
}

void require_status(cverl_status_t status, const char* context) {
  if (status != CVERL_OK) {
    std::cerr << context << " failed: " << cverl_status_string(status) << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  constexpr int64_t rows = 2;
  constexpr int64_t cols = 4;
  std::vector<float> old_lp{0.0f, 0.0f, 0.0f, 0.0f, 0.2f, -0.1f, 0.3f, -0.2f};
  std::vector<float> lp{0.1f, 0.4f, -0.2f, -0.6f, 0.0f, 0.2f, 0.6f, -0.3f};
  std::vector<float> adv{1.0f, 1.0f, -1.0f, -1.0f, 0.5f, -0.5f, 1.5f, -1.5f};
  std::vector<float> mask{1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};

  cverl_ppo_loss_result_t c_result{};
  require_status(
      cverl_ppo_clipped_loss_f32_cpu(
          ct(old_lp, rows, cols),
          ct(lp, rows, cols),
          ct(adv, rows, cols),
          ct(mask, rows, cols),
          0.2f,
          -1.0f,
          -1.0f,
          3.0f,
          CVERL_LOSS_AGG_TOKEN_MEAN,
          &c_result),
      "c ppo loss");

  torch::Tensor old_lp_t = torch::from_blob(old_lp.data(), {rows, cols}, torch::kFloat32).clone();
  torch::Tensor lp_t = torch::from_blob(lp.data(), {rows, cols}, torch::kFloat32).clone().set_requires_grad(true);
  torch::Tensor adv_t = torch::from_blob(adv.data(), {rows, cols}, torch::kFloat32).clone();
  torch::Tensor mask_t = torch::from_blob(mask.data(), {rows, cols}, torch::kFloat32).clone();

  auto t_result = cverl::torch_backend::ppo_clipped_loss(
      old_lp_t, lp_t, adv_t, mask_t, 0.2, -1.0, -1.0, 3.0, CVERL_LOSS_AGG_TOKEN_MEAN);
  require_close(t_result.pg_loss.item<float>(), c_result.pg_loss, 1.0e-6f, "torch ppo loss");
  require_close(t_result.pg_clipfrac.item<float>(), c_result.pg_clipfrac, 1.0e-6f, "torch ppo clipfrac");
  require_close(t_result.ppo_kl.item<float>(), c_result.ppo_kl, 1.0e-6f, "torch ppo kl");

  t_result.pg_loss.backward();
  std::vector<float> c_grad(old_lp.size());
  require_status(
      cverl_ppo_clipped_loss_backward_f32_cpu(
          ct(old_lp, rows, cols),
          ct(lp, rows, cols),
          ct(adv, rows, cols),
          ct(mask, rows, cols),
          0.2f,
          -1.0f,
          -1.0f,
          3.0f,
          CVERL_LOSS_AGG_TOKEN_MEAN,
          mt(c_grad, rows, cols)),
      "c ppo backward");
  torch::Tensor c_grad_t = torch::from_blob(c_grad.data(), {rows, cols}, torch::kFloat32);
  if (!torch::allclose(lp_t.grad(), c_grad_t, 1.0e-5, 1.0e-6)) {
    std::cerr << "torch ppo grad mismatch\n";
    std::cerr << "torch grad: " << lp_t.grad() << "\n";
    std::cerr << "c grad: " << c_grad_t << "\n";
    return 1;
  }

  torch::Tensor kl_logp = torch::tensor({{0.0f, -1.0f}, {-2.0f, 2.0f}}, torch::kFloat32).set_requires_grad(true);
  torch::Tensor kl_ref = torch::tensor({{-0.5f, -1.5f}, {-1.0f, 0.0f}}, torch::kFloat32);
  torch::Tensor kl = cverl::torch_backend::kl_penalty(kl_logp, kl_ref, CVERL_KL_K3);
  kl.sum().backward();
  std::vector<float> kl_logp_vec{0.0f, -1.0f, -2.0f, 2.0f};
  std::vector<float> kl_ref_vec{-0.5f, -1.5f, -1.0f, 0.0f};
  std::vector<float> kl_grad(4);
  require_status(
      cverl_kl_penalty_backward_f32_cpu(
          ct(kl_logp_vec, 2, 2), ct(kl_ref_vec, 2, 2), CVERL_KL_K3, mt(kl_grad, 2, 2)),
      "c kl backward");
  torch::Tensor kl_grad_t = torch::from_blob(kl_grad.data(), {2, 2}, torch::kFloat32);
  if (!torch::allclose(kl_logp.grad(), kl_grad_t, 1.0e-5, 1.0e-6)) {
    std::cerr << "torch kl grad mismatch\n";
    return 1;
  }

  std::cout << "cverl LibTorch backend tests passed\n";
  return 0;
}
