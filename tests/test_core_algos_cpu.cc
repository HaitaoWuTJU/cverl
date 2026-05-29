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
}

void test_kl() {
  std::vector<float> logp{0.0f, -1.0f, -2.0f, 2.0f};
  std::vector<float> ref{-0.5f, -1.5f, -1.0f, 0.0f};
  std::vector<float> out(4);
  require_status(cverl_kl_penalty_f32_cpu(ct(logp, 2, 2), ct(ref, 2, 2), CVERL_KL_K1, mt(out, 2, 2)), "kl k1");
  require_vec_close(out, {0.5f, 0.5f, -1.0f, 2.0f}, 1.0e-6f, "kl k1");

  require_status(cverl_kl_penalty_f32_cpu(ct(logp, 2, 2), ct(ref, 2, 2), CVERL_KL_K2, mt(out, 2, 2)), "kl k2");
  require_vec_close(out, {0.125f, 0.125f, 0.5f, 2.0f}, 1.0e-6f, "kl k2");
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

}  // namespace

int main() {
  test_masked_ops();
  test_kl();
  test_gae();
  test_grpo();
  test_ppo_loss();
  std::cout << "cverl CPU core algos tests passed\n";
  return 0;
}
