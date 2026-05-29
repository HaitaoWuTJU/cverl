#include "cverl/cverl.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr char kMagic[] = "CVERLGD1";
constexpr uint32_t kMinVersion = 1;
constexpr uint32_t kMaxVersion = 3;
constexpr float kAtol = 1.0e-5f;
constexpr float kRtol = 1.0e-5f;

enum RecordKind : uint32_t {
  kKindKl = 1,
  kKindGae = 2,
  kKindGrpo = 3,
  kKindPpo = 4,
};

template <typename T>
T read_scalar(std::ifstream& in) {
  T value{};
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!in) {
    std::cerr << "unexpected end of golden file\n";
    std::exit(1);
  }
  return value;
}

std::vector<float> read_tensor(std::ifstream& in, int64_t* rows, int64_t* cols) {
  *rows = static_cast<int64_t>(read_scalar<uint32_t>(in));
  *cols = static_cast<int64_t>(read_scalar<uint32_t>(in));
  if (*rows < 0 || *cols < 0) {
    std::cerr << "invalid tensor shape\n";
    std::exit(1);
  }
  std::vector<float> data(static_cast<size_t>(*rows * *cols));
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
  if (!in) {
    std::cerr << "unexpected end of tensor data\n";
    std::exit(1);
  }
  return data;
}

std::vector<int64_t> read_i64_array(std::ifstream& in) {
  uint32_t size = read_scalar<uint32_t>(in);
  std::vector<int64_t> data(size);
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(int64_t)));
  if (!in) {
    std::cerr << "unexpected end of int64 array\n";
    std::exit(1);
  }
  return data;
}

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

void require_shape(int64_t rows, int64_t cols, int64_t expected_rows, int64_t expected_cols, const std::string& name) {
  if (rows != expected_rows || cols != expected_cols) {
    std::cerr << name << " shape mismatch: expected [" << expected_rows << ", " << expected_cols
              << "] got [" << rows << ", " << cols << "]\n";
    std::exit(1);
  }
}

bool close_enough(float actual, float expected) {
  const float allowed = kAtol + kRtol * std::fabs(expected);
  return std::fabs(actual - expected) <= allowed;
}

void compare_tensor(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    const std::string& context) {
  if (actual.size() != expected.size()) {
    std::cerr << context << " size mismatch\n";
    std::exit(1);
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!close_enough(actual[i], expected[i])) {
      std::cerr << context << "[" << i << "] expected " << expected[i] << " got " << actual[i] << "\n";
      std::exit(1);
    }
  }
}

void compare_scalar(float actual, float expected, const std::string& context) {
  if (!close_enough(actual, expected)) {
    std::cerr << context << " expected " << expected << " got " << actual << "\n";
    std::exit(1);
  }
}

void read_expect_tensor_shape(
    std::ifstream& in,
    int64_t rows,
    int64_t cols,
    std::vector<float>* out,
    const std::string& name) {
  int64_t actual_rows = 0;
  int64_t actual_cols = 0;
  *out = read_tensor(in, &actual_rows, &actual_cols);
  require_shape(actual_rows, actual_cols, rows, cols, name);
}

void compare_kl(std::ifstream& in, uint32_t version) {
  uint32_t penalty = read_scalar<uint32_t>(in);
  int64_t rows = 0;
  int64_t cols = 0;
  std::vector<float> logp = read_tensor(in, &rows, &cols);
  std::vector<float> ref;
  std::vector<float> expected;
  read_expect_tensor_shape(in, rows, cols, &ref, "kl ref");
  read_expect_tensor_shape(in, rows, cols, &expected, "kl expected");

  std::vector<float> actual(expected.size());
  require_status(
      cverl_kl_penalty_f32_cpu(
          ct(logp, rows, cols),
          ct(ref, rows, cols),
          static_cast<cverl_kl_penalty_t>(penalty),
          mt(actual, rows, cols)),
      "kl");
  compare_tensor(actual, expected, "kl");

  if (version >= 3) {
    std::vector<float> expected_grad;
    read_expect_tensor_shape(in, rows, cols, &expected_grad, "kl expected grad");
    std::vector<float> actual_grad(expected_grad.size());
    require_status(
        cverl_kl_penalty_backward_f32_cpu(
            ct(logp, rows, cols),
            ct(ref, rows, cols),
            static_cast<cverl_kl_penalty_t>(penalty),
            mt(actual_grad, rows, cols)),
        "kl backward");
    compare_tensor(actual_grad, expected_grad, "kl grad_logprob");
  }
}

void compare_gae(std::ifstream& in) {
  float gamma = read_scalar<float>(in);
  float lam = read_scalar<float>(in);
  int64_t rows = 0;
  int64_t cols = 0;
  std::vector<float> rewards = read_tensor(in, &rows, &cols);
  std::vector<float> values;
  std::vector<float> mask;
  std::vector<float> expected_adv;
  std::vector<float> expected_ret;
  read_expect_tensor_shape(in, rows, cols, &values, "gae values");
  read_expect_tensor_shape(in, rows, cols, &mask, "gae mask");
  read_expect_tensor_shape(in, rows, cols, &expected_adv, "gae expected adv");
  read_expect_tensor_shape(in, rows, cols, &expected_ret, "gae expected ret");

  std::vector<float> actual_adv(expected_adv.size());
  std::vector<float> actual_ret(expected_ret.size());
  require_status(
      cverl_gae_advantage_return_f32_cpu(
          ct(rewards, rows, cols),
          ct(values, rows, cols),
          ct(mask, rows, cols),
          gamma,
          lam,
          mt(actual_adv, rows, cols),
          mt(actual_ret, rows, cols)),
      "gae");
  compare_tensor(actual_adv, expected_adv, "gae advantages");
  compare_tensor(actual_ret, expected_ret, "gae returns");
}

void compare_grpo(std::ifstream& in) {
  float epsilon = read_scalar<float>(in);
  uint32_t norm = read_scalar<uint32_t>(in);
  std::vector<int64_t> groups = read_i64_array(in);

  int64_t rows = 0;
  int64_t cols = 0;
  std::vector<float> rewards = read_tensor(in, &rows, &cols);
  if (groups.size() != static_cast<size_t>(rows)) {
    std::cerr << "grpo group id count mismatch\n";
    std::exit(1);
  }
  std::vector<float> mask;
  std::vector<float> expected_adv;
  std::vector<float> expected_ret;
  read_expect_tensor_shape(in, rows, cols, &mask, "grpo mask");
  read_expect_tensor_shape(in, rows, cols, &expected_adv, "grpo expected adv");
  read_expect_tensor_shape(in, rows, cols, &expected_ret, "grpo expected ret");

  std::vector<float> actual_adv(expected_adv.size());
  std::vector<float> actual_ret(expected_ret.size());
  require_status(
      cverl_grpo_outcome_advantage_f32_cpu(
          ct(rewards, rows, cols),
          ct(mask, rows, cols),
          groups.data(),
          epsilon,
          static_cast<int>(norm),
          mt(actual_adv, rows, cols),
          mt(actual_ret, rows, cols)),
      "grpo");
  compare_tensor(actual_adv, expected_adv, "grpo advantages");
  compare_tensor(actual_ret, expected_ret, "grpo returns");
}

void compare_ppo(std::ifstream& in, uint32_t version) {
  float clip = read_scalar<float>(in);
  float clip_low = read_scalar<float>(in);
  float clip_high = read_scalar<float>(in);
  float clip_c = read_scalar<float>(in);
  uint32_t agg = read_scalar<uint32_t>(in);
  int64_t rows = 0;
  int64_t cols = 0;
  std::vector<float> old_log_prob = read_tensor(in, &rows, &cols);
  std::vector<float> log_prob;
  std::vector<float> advantages;
  std::vector<float> mask;
  read_expect_tensor_shape(in, rows, cols, &log_prob, "ppo log_prob");
  read_expect_tensor_shape(in, rows, cols, &advantages, "ppo advantages");
  read_expect_tensor_shape(in, rows, cols, &mask, "ppo mask");

  const float expected_loss = read_scalar<float>(in);
  const float expected_clipfrac = read_scalar<float>(in);
  const float expected_kl = read_scalar<float>(in);
  const float expected_clipfrac_lower = read_scalar<float>(in);

  cverl_ppo_loss_result_t actual{};
  require_status(
      cverl_ppo_clipped_loss_f32_cpu(
          ct(old_log_prob, rows, cols),
          ct(log_prob, rows, cols),
          ct(advantages, rows, cols),
          ct(mask, rows, cols),
          clip,
          clip_low,
          clip_high,
          clip_c,
          static_cast<cverl_loss_agg_mode_t>(agg),
          &actual),
      "ppo");
  compare_scalar(actual.pg_loss, expected_loss, "ppo pg_loss");
  compare_scalar(actual.pg_clipfrac, expected_clipfrac, "ppo clipfrac");
  compare_scalar(actual.ppo_kl, expected_kl, "ppo kl");
  compare_scalar(actual.pg_clipfrac_lower, expected_clipfrac_lower, "ppo clipfrac_lower");

  if (version >= 2) {
    std::vector<float> expected_grad;
    read_expect_tensor_shape(in, rows, cols, &expected_grad, "ppo expected grad");
    std::vector<float> actual_grad(expected_grad.size());
    require_status(
        cverl_ppo_clipped_loss_backward_f32_cpu(
            ct(old_log_prob, rows, cols),
            ct(log_prob, rows, cols),
            ct(advantages, rows, cols),
            ct(mask, rows, cols),
            clip,
            clip_low,
            clip_high,
            clip_c,
            static_cast<cverl_loss_agg_mode_t>(agg),
            mt(actual_grad, rows, cols)),
        "ppo backward");
    compare_tensor(actual_grad, expected_grad, "ppo grad_log_prob");
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: compare_golden <golden.bin>\n";
    return 2;
  }

  std::ifstream in(argv[1], std::ios::binary);
  if (!in) {
    std::cerr << "failed to open " << argv[1] << "\n";
    return 1;
  }

  char magic[8]{};
  in.read(magic, sizeof(magic));
  if (!in || std::string(magic, sizeof(magic)) != std::string(kMagic, sizeof(magic))) {
    std::cerr << "invalid golden file magic\n";
    return 1;
  }
  uint32_t version = read_scalar<uint32_t>(in);
  if (version < kMinVersion || version > kMaxVersion) {
    std::cerr << "unsupported golden file version " << version << "\n";
    return 1;
  }

  uint32_t count = read_scalar<uint32_t>(in);
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t kind = read_scalar<uint32_t>(in);
    switch (kind) {
      case kKindKl:
        compare_kl(in, version);
        break;
      case kKindGae:
        compare_gae(in);
        break;
      case kKindGrpo:
        compare_grpo(in);
        break;
      case kKindPpo:
        compare_ppo(in, version);
        break;
      default:
        std::cerr << "unknown golden record kind " << kind << "\n";
        return 1;
    }
  }

  std::cout << "compared " << count << " golden records successfully\n";
  return 0;
}
