#include "cverl/cverl.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

cverl_const_tensor2d_t ct(const std::vector<float>& v, int64_t rows, int64_t cols) {
  return cverl_const_tensor2d_t{v.data(), CVERL_DTYPE_F32, CVERL_DEVICE_CPU, rows, cols};
}

cverl_tensor2d_t mt(std::vector<float>& v, int64_t rows, int64_t cols) {
  return cverl_tensor2d_t{v.data(), CVERL_DTYPE_F32, CVERL_DEVICE_CPU, rows, cols};
}

void require_status(cverl_status_t status, const std::string& context) {
  if (status != CVERL_OK) {
    std::cerr << context << " failed: " << cverl_status_string(status) << "\n";
    std::exit(1);
  }
}

void require_status_eq(cverl_status_t actual, cverl_status_t expected, const std::string& context) {
  if (actual != expected) {
    std::cerr << context << " expected status " << cverl_status_string(expected) << " got "
              << cverl_status_string(actual) << "\n";
    std::exit(1);
  }
}

void require_close(float actual, float expected, float atol, const std::string& context) {
  if (std::fabs(actual - expected) > atol) {
    std::cerr << context << " expected " << expected << " got " << actual << "\n";
    std::exit(1);
  }
}

void require_vec_close(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float atol,
    const std::string& context) {
  if (actual.size() != expected.size()) {
    std::cerr << context << " size mismatch\n";
    std::exit(1);
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (std::fabs(actual[i] - expected[i]) > atol) {
      std::cerr << context << "[" << i << "] expected " << expected[i] << " got " << actual[i] << "\n";
      std::exit(1);
    }
  }
}

void test_masked_ops() {
  std::vector<float> values{1.0f, 2.0f, 100.0f, 4.0f};
  std::vector<float> mask{1.0f, 1.0f, 0.0f, 1.0f};
  float sum = 0.0f;
  float mean = 0.0f;
  require_status(cverl_masked_sum_f32_cpu(ct(values, 2, 2), ct(mask, 2, 2), &sum), "masked_sum");
  require_status(cverl_masked_mean_f32_cpu(ct(values, 2, 2), ct(mask, 2, 2), &mean), "masked_mean");
  require_close(sum, 7.0f, 1.0e-6f, "masked_sum");
  require_close(mean, 7.0f / 3.0f, 1.0e-6f, "masked_mean");

  std::vector<float> whitened(4);
  require_status(cverl_masked_whiten_f32_cpu(ct(values, 2, 2), ct(mask, 2, 2), 1, mt(whitened, 2, 2)), "whiten");
  float whiten_mean = 0.0f;
  require_status(cverl_masked_mean_f32_cpu(ct(whitened, 2, 2), ct(mask, 2, 2), &whiten_mean), "whiten_mean");
  require_close(whiten_mean, 0.0f, 1.0e-5f, "whiten mean");

  std::vector<float> no_shift(4);
  require_status(
      cverl_masked_whiten_f32_cpu(ct(values, 2, 2), ct(mask, 2, 2), 0, mt(no_shift, 2, 2)),
      "whiten no shift");
  float no_shift_mean = 0.0f;
  require_status(cverl_masked_mean_f32_cpu(ct(no_shift, 2, 2), ct(mask, 2, 2), &no_shift_mean), "no shift mean");
  require_close(no_shift_mean, 7.0f / 3.0f, 1.0e-5f, "whiten no shift mean");
}

void test_kl() {
  std::vector<float> logp{0.0f, -1.0f, -2.0f, 2.0f};
  std::vector<float> ref{-0.5f, -1.5f, -1.0f, 0.0f};
  std::vector<float> out(4);
  require_status(cverl_kl_penalty_f32_cpu(ct(logp, 2, 2), ct(ref, 2, 2), CVERL_KL_K1, mt(out, 2, 2)), "kl k1");
  require_vec_close(out, {0.5f, 0.5f, -1.0f, 2.0f}, 1.0e-6f, "kl k1");

  require_status(cverl_kl_penalty_f32_cpu(ct(logp, 2, 2), ct(ref, 2, 2), CVERL_KL_ABS, mt(out, 2, 2)), "kl abs");
  require_vec_close(out, {0.5f, 0.5f, 1.0f, 2.0f}, 1.0e-6f, "kl abs");

  require_status(cverl_kl_penalty_f32_cpu(ct(logp, 2, 2), ct(ref, 2, 2), CVERL_KL_K2, mt(out, 2, 2)), "kl k2");
  require_vec_close(out, {0.125f, 0.125f, 0.5f, 2.0f}, 1.0e-6f, "kl k2");

  require_status(
      cverl_kl_penalty_backward_f32_cpu(ct(logp, 2, 2), ct(ref, 2, 2), CVERL_KL_K2, mt(out, 2, 2)),
      "kl k2 backward");
  require_vec_close(out, {0.5f, 0.5f, -1.0f, 2.0f}, 1.0e-6f, "kl k2 backward");

  require_status(
      cverl_kl_penalty_backward_f32_cpu(ct(logp, 2, 2), ct(ref, 2, 2), CVERL_KL_K3, mt(out, 2, 2)),
      "kl k3 backward");
  require_close(out[0], -(std::exp(-0.5f) - 1.0f), 1.0e-6f, "kl k3 backward");
}

void test_gae() {
  std::vector<float> rewards{
      0.0f, 0.1f, 0.2f, 1.0f,
      0.0f, 0.5f, 0.0f, 1.0f,
  };
  std::vector<float> values{
      0.0f, 0.2f, 0.3f, 0.4f,
      5.0f, 0.1f, 7.0f, 0.2f,
  };
  std::vector<float> mask{
      0.0f, 1.0f, 1.0f, 1.0f,
      0.0f, 1.0f, 0.0f, 1.0f,
  };
  std::vector<float> adv(8);
  std::vector<float> ret(8);
  require_status(
      cverl_gae_advantage_return_f32_cpu(
          ct(rewards, 2, 4), ct(values, 2, 4), ct(mask, 2, 4), 1.0f, 1.0f, mt(adv, 2, 4), mt(ret, 2, 4)),
      "gae");

  require_vec_close(ret, {1.1f, 1.3f, 1.2f, 1.0f, 6.4f, 1.5f, 7.8f, 1.0f}, 1.0e-5f, "gae returns");
  float mean = 0.0f;
  require_status(cverl_masked_mean_f32_cpu(ct(adv, 2, 4), ct(mask, 2, 4), &mean), "gae adv mean");
  require_close(mean, 0.0f, 1.0e-5f, "gae whitened adv mean");
}

void test_grpo() {
  std::vector<float> rewards{
      1.0f, 0.0f,
      3.0f, 0.0f,
      5.0f, 0.0f,
      9.0f, 0.0f,
  };
  std::vector<float> mask{
      1.0f, 1.0f,
      1.0f, 0.0f,
      1.0f, 1.0f,
      0.0f, 1.0f,
  };
  std::vector<int64_t> groups{0, 0, 1, 1};
  std::vector<float> adv(8);
  std::vector<float> ret(8);
  require_status(
      cverl_grpo_outcome_advantage_f32_cpu(
          ct(rewards, 4, 2), ct(mask, 4, 2), groups.data(), 1.0e-6f, 0, mt(adv, 4, 2), mt(ret, 4, 2)),
      "grpo");
  require_vec_close(adv, {-1.0f, -1.0f, 1.0f, 0.0f, -2.0f, -2.0f, 0.0f, 2.0f}, 1.0e-6f, "grpo adv");
  require_vec_close(ret, adv, 1.0e-6f, "grpo returns");

  std::vector<float> norm_rewards{1.0f, 0.0f, 3.0f, 0.0f};
  std::vector<float> norm_mask{1.0f, 0.0f, 1.0f, 1.0f};
  std::vector<int64_t> norm_groups{7, 7};
  std::vector<float> norm_adv(4);
  std::vector<float> norm_ret(4);
  require_status(
      cverl_grpo_outcome_advantage_f32_cpu(
          ct(norm_rewards, 2, 2),
          ct(norm_mask, 2, 2),
          norm_groups.data(),
          1.0e-6f,
          1,
          mt(norm_adv, 2, 2),
          mt(norm_ret, 2, 2)),
      "grpo norm");
  const float inv_std = 1.0f / (std::sqrt(2.0f) + 1.0e-6f);
  require_vec_close(norm_adv, {-inv_std, 0.0f, inv_std, inv_std}, 1.0e-5f, "grpo norm adv");
  require_vec_close(norm_ret, norm_adv, 1.0e-6f, "grpo norm returns");
}

void test_ppo_loss() {
  std::vector<float> old_lp{0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> lp{std::log(1.1f), std::log(1.5f), std::log(0.8f), std::log(0.5f)};
  std::vector<float> adv{1.0f, 1.0f, -1.0f, -1.0f};
  std::vector<float> mask{1.0f, 1.0f, 1.0f, 0.0f};
  cverl_ppo_loss_result_t result{};
  require_status(
      cverl_ppo_clipped_loss_f32_cpu(
          ct(old_lp, 2, 2),
          ct(lp, 2, 2),
          ct(adv, 2, 2),
          ct(mask, 2, 2),
          0.2f,
          -1.0f,
          -1.0f,
          3.0f,
          CVERL_LOSS_AGG_TOKEN_MEAN,
          &result),
      "ppo loss");
  require_close(result.pg_loss, (-1.1f - 1.2f + 0.8f) / 3.0f, 1.0e-5f, "ppo pg_loss");
  require_close(result.pg_clipfrac, 1.0f / 3.0f, 1.0e-6f, "ppo clipfrac");
}

void test_ppo_aggregation_modes() {
  std::vector<float> old_lp(4, 0.0f);
  std::vector<float> lp(4, 0.0f);
  std::vector<float> adv{1.0f, 2.0f, 3.0f, 100.0f};
  std::vector<float> mask{1.0f, 1.0f, 1.0f, 0.0f};
  cverl_ppo_loss_result_t result{};

  require_status(
      cverl_ppo_clipped_loss_f32_cpu(
          ct(old_lp, 2, 2),
          ct(lp, 2, 2),
          ct(adv, 2, 2),
          ct(mask, 2, 2),
          0.2f,
          -1.0f,
          -1.0f,
          3.0f,
          CVERL_LOSS_AGG_TOKEN_MEAN,
          &result),
      "ppo token mean");
  require_close(result.pg_loss, -2.0f, 1.0e-6f, "ppo token mean loss");

  require_status(
      cverl_ppo_clipped_loss_f32_cpu(
          ct(old_lp, 2, 2),
          ct(lp, 2, 2),
          ct(adv, 2, 2),
          ct(mask, 2, 2),
          0.2f,
          -1.0f,
          -1.0f,
          3.0f,
          CVERL_LOSS_AGG_SEQ_MEAN_TOKEN_SUM,
          &result),
      "ppo seq mean token sum");
  require_close(result.pg_loss, -3.0f, 1.0e-6f, "ppo seq mean token sum loss");

  require_status(
      cverl_ppo_clipped_loss_f32_cpu(
          ct(old_lp, 2, 2),
          ct(lp, 2, 2),
          ct(adv, 2, 2),
          ct(mask, 2, 2),
          0.2f,
          -1.0f,
          -1.0f,
          3.0f,
          CVERL_LOSS_AGG_SEQ_MEAN_TOKEN_MEAN,
          &result),
      "ppo seq mean token mean");
  require_close(result.pg_loss, -2.25f, 1.0e-6f, "ppo seq mean token mean loss");
}

void test_ppo_loss_backward() {
  std::vector<float> old_lp{0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> lp{std::log(1.1f), std::log(1.5f), std::log(0.9f), std::log(0.5f)};
  std::vector<float> adv{1.0f, 1.0f, -1.0f, -1.0f};
  std::vector<float> mask{1.0f, 1.0f, 1.0f, 0.0f};
  std::vector<float> grad(4);
  require_status(
      cverl_ppo_clipped_loss_backward_f32_cpu(
          ct(old_lp, 2, 2),
          ct(lp, 2, 2),
          ct(adv, 2, 2),
          ct(mask, 2, 2),
          0.2f,
          -1.0f,
          -1.0f,
          3.0f,
          CVERL_LOSS_AGG_TOKEN_MEAN,
          mt(grad, 2, 2)),
      "ppo backward");
  require_vec_close(grad, {-1.1f / 3.0f, 0.0f, 0.9f / 3.0f, 0.0f}, 1.0e-6f, "ppo grad");
}

void test_invalid_arguments() {
  std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> short_mask{1.0f, 1.0f};
  std::vector<float> out(4);
  float scalar = 0.0f;

  require_status_eq(
      cverl_masked_sum_f32_cpu(ct(values, 2, 2), ct(short_mask, 1, 2), &scalar),
      CVERL_ERR_INVALID_ARGUMENT,
      "masked_sum shape validation");
  require_status_eq(
      cverl_masked_mean_f32_cpu(ct(values, 2, 2), ct(values, 2, 2), nullptr),
      CVERL_ERR_INVALID_ARGUMENT,
      "masked_mean null output validation");
  require_status_eq(
      cverl_kl_penalty_f32_cpu(
          ct(values, 2, 2), ct(values, 2, 2), static_cast<cverl_kl_penalty_t>(99), mt(out, 2, 2)),
      CVERL_ERR_INVALID_ARGUMENT,
      "kl invalid mode validation");
  require_status_eq(
      cverl_gae_advantage_return_f32_cpu(
          ct(values, 2, 2), ct(values, 2, 2), ct(short_mask, 1, 2), 1.0f, 1.0f, mt(out, 2, 2), mt(out, 2, 2)),
      CVERL_ERR_INVALID_ARGUMENT,
      "gae shape validation");
  require_status_eq(
      cverl_grpo_outcome_advantage_f32_cpu(
          ct(values, 2, 2), ct(values, 2, 2), nullptr, 1.0e-6f, 0, mt(out, 2, 2), mt(out, 2, 2)),
      CVERL_ERR_INVALID_ARGUMENT,
      "grpo null group validation");

  cverl_ppo_loss_result_t result{};
  require_status_eq(
      cverl_ppo_clipped_loss_f32_cpu(
          ct(values, 2, 2),
          ct(values, 2, 2),
          ct(values, 2, 2),
          ct(values, 2, 2),
          0.2f,
          -1.0f,
          -1.0f,
          1.0f,
          CVERL_LOSS_AGG_TOKEN_MEAN,
          &result),
      CVERL_ERR_INVALID_ARGUMENT,
      "ppo clip ratio c validation");
  require_status_eq(
      cverl_ppo_clipped_loss_backward_f32_cpu(
          ct(values, 2, 2),
          ct(short_mask, 1, 2),
          ct(values, 2, 2),
          ct(values, 2, 2),
          0.2f,
          -1.0f,
          -1.0f,
          3.0f,
          CVERL_LOSS_AGG_TOKEN_MEAN,
          mt(out, 2, 2)),
      CVERL_ERR_INVALID_ARGUMENT,
      "ppo backward shape validation");
}

}  // namespace

int main() {
  test_masked_ops();
  test_kl();
  test_gae();
  test_grpo();
  test_ppo_loss();
  test_ppo_aggregation_modes();
  test_ppo_loss_backward();
  test_invalid_arguments();
  std::cout << "cverl C ABI core algos tests passed\n";
  return 0;
}
