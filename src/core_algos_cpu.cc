#include "cverl/core_algos.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace {

constexpr float kEps = 1.0e-8f;

const float* ptr(cverl_const_tensor2d_t t) {
  return static_cast<const float*>(t.data);
}

float* ptr(cverl_tensor2d_t t) {
  return static_cast<float*>(t.data);
}

bool valid_const_f32_cpu(cverl_const_tensor2d_t t) {
  return t.data != nullptr && t.dtype == CVERL_DTYPE_F32 && t.device == CVERL_DEVICE_CPU &&
         t.rows >= 0 && t.cols >= 0;
}

bool valid_f32_cpu(cverl_tensor2d_t t) {
  return t.data != nullptr && t.dtype == CVERL_DTYPE_F32 && t.device == CVERL_DEVICE_CPU &&
         t.rows >= 0 && t.cols >= 0;
}

bool same_shape(cverl_const_tensor2d_t a, cverl_const_tensor2d_t b) {
  return a.rows == b.rows && a.cols == b.cols;
}

bool same_shape(cverl_const_tensor2d_t a, cverl_tensor2d_t b) {
  return a.rows == b.rows && a.cols == b.cols;
}

cverl_status_t validate_binary(cverl_const_tensor2d_t a, cverl_const_tensor2d_t b) {
  if (!valid_const_f32_cpu(a) || !valid_const_f32_cpu(b) || !same_shape(a, b)) {
    return CVERL_ERR_INVALID_ARGUMENT;
  }
  return CVERL_OK;
}

cverl_status_t validate_out(cverl_const_tensor2d_t reference, cverl_tensor2d_t out) {
  if (!valid_f32_cpu(out) || !same_shape(reference, out)) {
    return CVERL_ERR_INVALID_ARGUMENT;
  }
  return CVERL_OK;
}

int64_t numel(cverl_const_tensor2d_t t) {
  return t.rows * t.cols;
}

float masked_sum_raw(const float* values, const float* mask, int64_t n) {
  double acc = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    if (mask[i] != 0.0f) {
      acc += static_cast<double>(values[i]) * static_cast<double>(mask[i]);
    }
  }
  return static_cast<float>(acc);
}

float mask_sum_raw(const float* mask, int64_t n) {
  double acc = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    acc += static_cast<double>(mask[i]);
  }
  return static_cast<float>(acc);
}

float masked_mean_raw(const float* values, const float* mask, int64_t n) {
  return masked_sum_raw(values, mask, n) / (mask_sum_raw(mask, n) + kEps);
}

float masked_var_raw(const float* values, const float* mask, int64_t n) {
  const float denom = mask_sum_raw(mask, n);
  if (denom <= 1.0f) {
    return NAN;
  }
  const float mean = masked_mean_raw(values, mask, n);
  double acc = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    if (mask[i] != 0.0f) {
      const double centered = static_cast<double>(values[i]) - static_cast<double>(mean);
      acc += centered * centered * static_cast<double>(mask[i]);
    }
  }
  const double biased = acc / (static_cast<double>(denom) + static_cast<double>(kEps));
  return static_cast<float>(biased * (static_cast<double>(denom) / static_cast<double>(denom - 1.0f)));
}

float aggregate_loss(
    const std::vector<float>& losses,
    const float* mask,
    int64_t rows,
    int64_t cols,
    cverl_loss_agg_mode_t mode) {
  if (mode == CVERL_LOSS_AGG_TOKEN_MEAN) {
    return masked_sum_raw(losses.data(), mask, rows * cols) / (mask_sum_raw(mask, rows * cols) + kEps);
  }

  double seq_acc = 0.0;
  double seq_count = 0.0;
  for (int64_t r = 0; r < rows; ++r) {
    double token_sum = 0.0;
    double token_count = 0.0;
    for (int64_t c = 0; c < cols; ++c) {
      const int64_t i = r * cols + c;
      if (mask[i] != 0.0f) {
        token_sum += static_cast<double>(losses[i]) * static_cast<double>(mask[i]);
        token_count += static_cast<double>(mask[i]);
      }
    }
    if (token_count > 0.0) {
      seq_acc += mode == CVERL_LOSS_AGG_SEQ_MEAN_TOKEN_MEAN ? token_sum / token_count : token_sum;
      seq_count += 1.0;
    }
  }
  return static_cast<float>(seq_acc / (seq_count + static_cast<double>(kEps)));
}

}  // namespace

extern "C" cverl_status_t cverl_masked_sum_f32_cpu(
    cverl_const_tensor2d_t values,
    cverl_const_tensor2d_t mask,
    float* out_sum) {
  if (out_sum == nullptr) {
    return CVERL_ERR_INVALID_ARGUMENT;
  }
  cverl_status_t status = validate_binary(values, mask);
  if (status != CVERL_OK) {
    return status;
  }
  *out_sum = masked_sum_raw(ptr(values), ptr(mask), numel(values));
  return CVERL_OK;
}

extern "C" cverl_status_t cverl_masked_mean_f32_cpu(
    cverl_const_tensor2d_t values,
    cverl_const_tensor2d_t mask,
    float* out_mean) {
  if (out_mean == nullptr) {
    return CVERL_ERR_INVALID_ARGUMENT;
  }
  cverl_status_t status = validate_binary(values, mask);
  if (status != CVERL_OK) {
    return status;
  }
  *out_mean = masked_mean_raw(ptr(values), ptr(mask), numel(values));
  return CVERL_OK;
}

extern "C" cverl_status_t cverl_masked_whiten_f32_cpu(
    cverl_const_tensor2d_t values,
    cverl_const_tensor2d_t mask,
    int shift_mean,
    cverl_tensor2d_t out) {
  cverl_status_t status = validate_binary(values, mask);
  if (status != CVERL_OK) {
    return status;
  }
  status = validate_out(values, out);
  if (status != CVERL_OK) {
    return status;
  }

  const int64_t n = numel(values);
  const float mean = masked_mean_raw(ptr(values), ptr(mask), n);
  const float var = masked_var_raw(ptr(values), ptr(mask), n);
  const float inv_std = 1.0f / std::sqrt(var + kEps);
  const float add_back = shift_mean ? 0.0f : mean;
  float* dst = ptr(out);
  for (int64_t i = 0; i < n; ++i) {
    dst[i] = (ptr(values)[i] - mean) * inv_std + add_back;
  }
  return CVERL_OK;
}

extern "C" cverl_status_t cverl_kl_penalty_f32_cpu(
    cverl_const_tensor2d_t logprob,
    cverl_const_tensor2d_t ref_logprob,
    cverl_kl_penalty_t penalty,
    cverl_tensor2d_t out) {
  cverl_status_t status = validate_binary(logprob, ref_logprob);
  if (status != CVERL_OK) {
    return status;
  }
  status = validate_out(logprob, out);
  if (status != CVERL_OK) {
    return status;
  }

  const int64_t n = numel(logprob);
  const float* lp = ptr(logprob);
  const float* ref = ptr(ref_logprob);
  float* dst = ptr(out);
  for (int64_t i = 0; i < n; ++i) {
    const float diff = lp[i] - ref[i];
    switch (penalty) {
      case CVERL_KL_K1:
        dst[i] = diff;
        break;
      case CVERL_KL_ABS:
        dst[i] = std::fabs(diff);
        break;
      case CVERL_KL_K2:
        dst[i] = 0.5f * diff * diff;
        break;
      case CVERL_KL_K3: {
        const float kl = std::clamp(ref[i] - lp[i], -20.0f, 20.0f);
        dst[i] = std::clamp(std::exp(kl) - kl - 1.0f, -10.0f, 10.0f);
        break;
      }
      default:
        return CVERL_ERR_UNSUPPORTED;
    }
  }
  return CVERL_OK;
}

extern "C" cverl_status_t cverl_gae_advantage_return_f32_cpu(
    cverl_const_tensor2d_t token_level_rewards,
    cverl_const_tensor2d_t values,
    cverl_const_tensor2d_t response_mask,
    float gamma,
    float lam,
    cverl_tensor2d_t advantages,
    cverl_tensor2d_t returns) {
  cverl_status_t status = validate_binary(token_level_rewards, values);
  if (status != CVERL_OK) {
    return status;
  }
  status = validate_binary(token_level_rewards, response_mask);
  if (status != CVERL_OK) {
    return status;
  }
  status = validate_out(token_level_rewards, advantages);
  if (status != CVERL_OK) {
    return status;
  }
  status = validate_out(token_level_rewards, returns);
  if (status != CVERL_OK) {
    return status;
  }

  const int64_t rows = token_level_rewards.rows;
  const int64_t cols = token_level_rewards.cols;
  const float* rewards = ptr(token_level_rewards);
  const float* vals = ptr(values);
  const float* mask = ptr(response_mask);
  float* adv = ptr(advantages);
  float* ret = ptr(returns);

  for (int64_t r = 0; r < rows; ++r) {
    float next_value = 0.0f;
    float last_gae_lam = 0.0f;
    for (int64_t c = cols - 1; c >= 0; --c) {
      const int64_t i = r * cols + c;
      const float delta = rewards[i] + gamma * next_value - vals[i];
      const float candidate = delta + gamma * lam * last_gae_lam;
      next_value = vals[i] * mask[i] + (1.0f - mask[i]) * next_value;
      last_gae_lam = candidate * mask[i] + (1.0f - mask[i]) * last_gae_lam;
      adv[i] = last_gae_lam;
    }
  }
  for (int64_t i = 0; i < rows * cols; ++i) {
    ret[i] = adv[i] + vals[i];
  }

  cverl_const_tensor2d_t adv_const{adv, CVERL_DTYPE_F32, CVERL_DEVICE_CPU, rows, cols};
  return cverl_masked_whiten_f32_cpu(adv_const, response_mask, 1, advantages);
}

extern "C" cverl_status_t cverl_grpo_outcome_advantage_f32_cpu(
    cverl_const_tensor2d_t token_level_rewards,
    cverl_const_tensor2d_t response_mask,
    const int64_t* group_ids,
    float epsilon,
    int norm_adv_by_std,
    cverl_tensor2d_t advantages,
    cverl_tensor2d_t returns) {
  if (group_ids == nullptr) {
    return CVERL_ERR_INVALID_ARGUMENT;
  }
  cverl_status_t status = validate_binary(token_level_rewards, response_mask);
  if (status != CVERL_OK) {
    return status;
  }
  status = validate_out(token_level_rewards, advantages);
  if (status != CVERL_OK) {
    return status;
  }
  status = validate_out(token_level_rewards, returns);
  if (status != CVERL_OK) {
    return status;
  }

  const int64_t rows = token_level_rewards.rows;
  const int64_t cols = token_level_rewards.cols;
  const float* rewards = ptr(token_level_rewards);
  const float* mask = ptr(response_mask);
  float* adv = ptr(advantages);
  float* ret = ptr(returns);

  std::vector<float> scores(rows, 0.0f);
  std::unordered_map<int64_t, std::vector<int64_t>> groups;
  for (int64_t r = 0; r < rows; ++r) {
    double acc = 0.0;
    for (int64_t c = 0; c < cols; ++c) {
      acc += static_cast<double>(rewards[r * cols + c]);
    }
    scores[r] = static_cast<float>(acc);
    groups[group_ids[r]].push_back(r);
  }

  std::unordered_map<int64_t, float> means;
  std::unordered_map<int64_t, float> stds;
  for (const auto& entry : groups) {
    const std::vector<int64_t>& ids = entry.second;
    if (ids.size() == 1) {
      means[entry.first] = 0.0f;
      stds[entry.first] = 1.0f;
      continue;
    }
    double sum = 0.0;
    for (int64_t id : ids) {
      sum += scores[id];
    }
    const double mean = sum / static_cast<double>(ids.size());
    double sq = 0.0;
    for (int64_t id : ids) {
      const double centered = static_cast<double>(scores[id]) - mean;
      sq += centered * centered;
    }
    means[entry.first] = static_cast<float>(mean);
    stds[entry.first] = static_cast<float>(std::sqrt(sq / static_cast<double>(ids.size() - 1)));
  }

  for (int64_t r = 0; r < rows; ++r) {
    float scalar = scores[r] - means[group_ids[r]];
    if (norm_adv_by_std) {
      scalar /= (stds[group_ids[r]] + epsilon);
    }
    for (int64_t c = 0; c < cols; ++c) {
      const int64_t i = r * cols + c;
      adv[i] = scalar * mask[i];
      ret[i] = adv[i];
    }
  }
  return CVERL_OK;
}

extern "C" cverl_status_t cverl_ppo_clipped_loss_f32_cpu(
    cverl_const_tensor2d_t old_log_prob,
    cverl_const_tensor2d_t log_prob,
    cverl_const_tensor2d_t advantages,
    cverl_const_tensor2d_t response_mask,
    float clip_ratio,
    float clip_ratio_low,
    float clip_ratio_high,
    float clip_ratio_c,
    cverl_loss_agg_mode_t agg_mode,
    cverl_ppo_loss_result_t* out) {
  if (out == nullptr || clip_ratio_c <= 1.0f) {
    return CVERL_ERR_INVALID_ARGUMENT;
  }
  if (clip_ratio_low < 0.0f) {
    clip_ratio_low = clip_ratio;
  }
  if (clip_ratio_high < 0.0f) {
    clip_ratio_high = clip_ratio;
  }
  cverl_status_t status = validate_binary(old_log_prob, log_prob);
  if (status != CVERL_OK) {
    return status;
  }
  status = validate_binary(old_log_prob, advantages);
  if (status != CVERL_OK) {
    return status;
  }
  status = validate_binary(old_log_prob, response_mask);
  if (status != CVERL_OK) {
    return status;
  }

  const int64_t n = numel(old_log_prob);
  const float* old_lp = ptr(old_log_prob);
  const float* lp = ptr(log_prob);
  const float* adv = ptr(advantages);
  const float* mask = ptr(response_mask);
  std::vector<float> losses(n);
  std::vector<float> clipfrac(n);
  std::vector<float> ppo_kl(n);
  std::vector<float> clipfrac_lower(n);

  for (int64_t i = 0; i < n; ++i) {
    const float neg_approx_kl = std::clamp(lp[i] - old_lp[i], -20.0f, 20.0f);
    const float ratio = std::exp(neg_approx_kl);
    ppo_kl[i] = -neg_approx_kl;

    const float loss1 = -adv[i] * ratio;
    const float clipped_ratio = std::clamp(ratio, 1.0f - clip_ratio_low, 1.0f + clip_ratio_high);
    const float loss2 = -adv[i] * clipped_ratio;
    const float clipped_loss1 = std::max(loss1, loss2);
    clipfrac[i] = loss2 > loss1 ? 1.0f : 0.0f;

    const float loss3 = -adv[i] * clip_ratio_c;
    const float clipped_loss2 = std::min(loss3, clipped_loss1);
    clipfrac_lower[i] = (clipped_loss1 > loss3 && adv[i] < 0.0f) ? 1.0f : 0.0f;
    losses[i] = adv[i] < 0.0f ? clipped_loss2 : clipped_loss1;
  }

  out->pg_loss = aggregate_loss(losses, mask, old_log_prob.rows, old_log_prob.cols, agg_mode);
  out->pg_clipfrac = masked_mean_raw(clipfrac.data(), mask, n);
  out->ppo_kl = masked_mean_raw(ppo_kl.data(), mask, n);
  out->pg_clipfrac_lower = masked_mean_raw(clipfrac_lower.data(), mask, n);
  return CVERL_OK;
}
