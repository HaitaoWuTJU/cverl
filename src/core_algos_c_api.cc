#include "cverl/core_algos.h"

#include <torch/torch.h>

#include <exception>

#include "cverl/torch/core_algos_torch.h"

namespace {

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

torch::Tensor as_tensor(cverl_const_tensor2d_t t) {
  return torch::from_blob(const_cast<void*>(t.data), {t.rows, t.cols}, torch::kFloat32);
}

torch::Tensor as_tensor(cverl_tensor2d_t t) {
  return torch::from_blob(t.data, {t.rows, t.cols}, torch::kFloat32);
}

void copy_to(torch::Tensor src, cverl_tensor2d_t dst) {
  torch::NoGradGuard no_grad;
  as_tensor(dst).copy_(src.to(torch::kCPU).contiguous());
}

}  // namespace

extern "C" cverl_status_t cverl_masked_sum_f32_cpu(
    cverl_const_tensor2d_t values,
    cverl_const_tensor2d_t mask,
    float* out_sum) {
  if (out_sum == nullptr || validate_binary(values, mask) != CVERL_OK) {
    return CVERL_ERR_INVALID_ARGUMENT;
  }
  try {
    *out_sum = cverl::torch_backend::masked_sum(as_tensor(values), as_tensor(mask)).item<float>();
    return CVERL_OK;
  } catch (const std::exception&) {
    return CVERL_ERR_INVALID_ARGUMENT;
  }
}

extern "C" cverl_status_t cverl_masked_mean_f32_cpu(
    cverl_const_tensor2d_t values,
    cverl_const_tensor2d_t mask,
    float* out_mean) {
  if (out_mean == nullptr || validate_binary(values, mask) != CVERL_OK) {
    return CVERL_ERR_INVALID_ARGUMENT;
  }
  try {
    *out_mean = cverl::torch_backend::masked_mean(as_tensor(values), as_tensor(mask)).item<float>();
    return CVERL_OK;
  } catch (const std::exception&) {
    return CVERL_ERR_INVALID_ARGUMENT;
  }
}

extern "C" cverl_status_t cverl_masked_whiten_f32_cpu(
    cverl_const_tensor2d_t values,
    cverl_const_tensor2d_t mask,
    int shift_mean,
    cverl_tensor2d_t out) {
  if (validate_binary(values, mask) != CVERL_OK || validate_out(values, out) != CVERL_OK) {
    return CVERL_ERR_INVALID_ARGUMENT;
  }
  try {
    copy_to(cverl::torch_backend::masked_whiten(as_tensor(values), as_tensor(mask), shift_mean != 0), out);
    return CVERL_OK;
  } catch (const std::exception&) {
    return CVERL_ERR_INVALID_ARGUMENT;
  }
}

extern "C" cverl_status_t cverl_kl_penalty_f32_cpu(
    cverl_const_tensor2d_t logprob,
    cverl_const_tensor2d_t ref_logprob,
    cverl_kl_penalty_t penalty,
    cverl_tensor2d_t out) {
  if (validate_binary(logprob, ref_logprob) != CVERL_OK || validate_out(logprob, out) != CVERL_OK) {
    return CVERL_ERR_INVALID_ARGUMENT;
  }
  try {
    copy_to(cverl::torch_backend::kl_penalty(as_tensor(logprob), as_tensor(ref_logprob), penalty), out);
    return CVERL_OK;
  } catch (const std::exception&) {
    return CVERL_ERR_INVALID_ARGUMENT;
  }
}

extern "C" cverl_status_t cverl_kl_penalty_backward_f32_cpu(
    cverl_const_tensor2d_t logprob,
    cverl_const_tensor2d_t ref_logprob,
    cverl_kl_penalty_t penalty,
    cverl_tensor2d_t grad_logprob) {
  if (validate_binary(logprob, ref_logprob) != CVERL_OK || validate_out(logprob, grad_logprob) != CVERL_OK) {
    return CVERL_ERR_INVALID_ARGUMENT;
  }
  try {
    torch::Tensor logp = as_tensor(logprob).clone().set_requires_grad(true);
    torch::Tensor ref = as_tensor(ref_logprob).clone();
    cverl::torch_backend::kl_penalty(logp, ref, penalty).sum().backward();
    copy_to(logp.grad(), grad_logprob);
    return CVERL_OK;
  } catch (const std::exception&) {
    return CVERL_ERR_INVALID_ARGUMENT;
  }
}

extern "C" cverl_status_t cverl_gae_advantage_return_f32_cpu(
    cverl_const_tensor2d_t token_level_rewards,
    cverl_const_tensor2d_t values,
    cverl_const_tensor2d_t response_mask,
    float gamma,
    float lam,
    cverl_tensor2d_t advantages,
    cverl_tensor2d_t returns) {
  if (validate_binary(token_level_rewards, values) != CVERL_OK ||
      validate_binary(token_level_rewards, response_mask) != CVERL_OK ||
      validate_out(token_level_rewards, advantages) != CVERL_OK ||
      validate_out(token_level_rewards, returns) != CVERL_OK) {
    return CVERL_ERR_INVALID_ARGUMENT;
  }
  try {
    torch::Tensor ret;
    torch::Tensor adv = cverl::torch_backend::gae_advantage_return(
        as_tensor(token_level_rewards), as_tensor(values), as_tensor(response_mask), gamma, lam, &ret);
    copy_to(adv, advantages);
    copy_to(ret, returns);
    return CVERL_OK;
  } catch (const std::exception&) {
    return CVERL_ERR_INVALID_ARGUMENT;
  }
}

extern "C" cverl_status_t cverl_grpo_outcome_advantage_f32_cpu(
    cverl_const_tensor2d_t token_level_rewards,
    cverl_const_tensor2d_t response_mask,
    const int64_t* group_ids,
    float epsilon,
    int norm_adv_by_std,
    cverl_tensor2d_t advantages,
    cverl_tensor2d_t returns) {
  if (group_ids == nullptr || validate_binary(token_level_rewards, response_mask) != CVERL_OK ||
      validate_out(token_level_rewards, advantages) != CVERL_OK ||
      validate_out(token_level_rewards, returns) != CVERL_OK) {
    return CVERL_ERR_INVALID_ARGUMENT;
  }
  try {
    torch::Tensor groups = torch::from_blob(const_cast<int64_t*>(group_ids), {token_level_rewards.rows}, torch::kLong);
    torch::Tensor ret;
    torch::Tensor adv = cverl::torch_backend::grpo_outcome_advantage(
        as_tensor(token_level_rewards),
        as_tensor(response_mask),
        groups,
        epsilon,
        norm_adv_by_std != 0,
        &ret);
    copy_to(adv, advantages);
    copy_to(ret, returns);
    return CVERL_OK;
  } catch (const std::exception&) {
    return CVERL_ERR_INVALID_ARGUMENT;
  }
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
  if (out == nullptr || validate_binary(old_log_prob, log_prob) != CVERL_OK ||
      validate_binary(old_log_prob, advantages) != CVERL_OK ||
      validate_binary(old_log_prob, response_mask) != CVERL_OK) {
    return CVERL_ERR_INVALID_ARGUMENT;
  }
  try {
    auto result = cverl::torch_backend::ppo_clipped_loss(
        as_tensor(old_log_prob),
        as_tensor(log_prob),
        as_tensor(advantages),
        as_tensor(response_mask),
        clip_ratio,
        clip_ratio_low,
        clip_ratio_high,
        clip_ratio_c,
        agg_mode);
    out->pg_loss = result.pg_loss.item<float>();
    out->pg_clipfrac = result.pg_clipfrac.item<float>();
    out->ppo_kl = result.ppo_kl.item<float>();
    out->pg_clipfrac_lower = result.pg_clipfrac_lower.item<float>();
    return CVERL_OK;
  } catch (const std::exception&) {
    return CVERL_ERR_INVALID_ARGUMENT;
  }
}

extern "C" cverl_status_t cverl_ppo_clipped_loss_backward_f32_cpu(
    cverl_const_tensor2d_t old_log_prob,
    cverl_const_tensor2d_t log_prob,
    cverl_const_tensor2d_t advantages,
    cverl_const_tensor2d_t response_mask,
    float clip_ratio,
    float clip_ratio_low,
    float clip_ratio_high,
    float clip_ratio_c,
    cverl_loss_agg_mode_t agg_mode,
    cverl_tensor2d_t grad_log_prob) {
  if (validate_binary(old_log_prob, log_prob) != CVERL_OK ||
      validate_binary(old_log_prob, advantages) != CVERL_OK ||
      validate_binary(old_log_prob, response_mask) != CVERL_OK ||
      validate_out(old_log_prob, grad_log_prob) != CVERL_OK) {
    return CVERL_ERR_INVALID_ARGUMENT;
  }
  try {
    torch::Tensor lp = as_tensor(log_prob).clone().set_requires_grad(true);
    auto result = cverl::torch_backend::ppo_clipped_loss(
        as_tensor(old_log_prob).clone(),
        lp,
        as_tensor(advantages).clone(),
        as_tensor(response_mask).clone(),
        clip_ratio,
        clip_ratio_low,
        clip_ratio_high,
        clip_ratio_c,
        agg_mode);
    result.pg_loss.backward();
    copy_to(lp.grad(), grad_log_prob);
    return CVERL_OK;
  } catch (const std::exception&) {
    return CVERL_ERR_INVALID_ARGUMENT;
  }
}
