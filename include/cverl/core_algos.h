#pragma once

#include <stdint.h>

#include "cverl/status.h"
#include "cverl/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  CVERL_LOSS_AGG_TOKEN_MEAN = 0,
  CVERL_LOSS_AGG_SEQ_MEAN_TOKEN_SUM = 1,
  CVERL_LOSS_AGG_SEQ_MEAN_TOKEN_MEAN = 2,
} cverl_loss_agg_mode_t;

typedef enum {
  CVERL_KL_K1 = 0,
  CVERL_KL_ABS = 1,
  CVERL_KL_K2 = 2,
  CVERL_KL_K3 = 3,
} cverl_kl_penalty_t;

typedef struct {
  float pg_loss;
  float pg_clipfrac;
  float ppo_kl;
  float pg_clipfrac_lower;
} cverl_ppo_loss_result_t;

cverl_status_t cverl_masked_sum_f32_cpu(
    cverl_const_tensor2d_t values,
    cverl_const_tensor2d_t mask,
    float* out_sum);

cverl_status_t cverl_masked_mean_f32_cpu(
    cverl_const_tensor2d_t values,
    cverl_const_tensor2d_t mask,
    float* out_mean);

cverl_status_t cverl_masked_whiten_f32_cpu(
    cverl_const_tensor2d_t values,
    cverl_const_tensor2d_t mask,
    int shift_mean,
    cverl_tensor2d_t out);

cverl_status_t cverl_kl_penalty_f32_cpu(
    cverl_const_tensor2d_t logprob,
    cverl_const_tensor2d_t ref_logprob,
    cverl_kl_penalty_t penalty,
    cverl_tensor2d_t out);

cverl_status_t cverl_gae_advantage_return_f32_cpu(
    cverl_const_tensor2d_t token_level_rewards,
    cverl_const_tensor2d_t values,
    cverl_const_tensor2d_t response_mask,
    float gamma,
    float lam,
    cverl_tensor2d_t advantages,
    cverl_tensor2d_t returns);

cverl_status_t cverl_grpo_outcome_advantage_f32_cpu(
    cverl_const_tensor2d_t token_level_rewards,
    cverl_const_tensor2d_t response_mask,
    const int64_t* group_ids,
    float epsilon,
    int norm_adv_by_std,
    cverl_tensor2d_t advantages,
    cverl_tensor2d_t returns);

cverl_status_t cverl_ppo_clipped_loss_f32_cpu(
    cverl_const_tensor2d_t old_log_prob,
    cverl_const_tensor2d_t log_prob,
    cverl_const_tensor2d_t advantages,
    cverl_const_tensor2d_t response_mask,
    float clip_ratio,
    float clip_ratio_low,
    float clip_ratio_high,
    float clip_ratio_c,
    cverl_loss_agg_mode_t agg_mode,
    cverl_ppo_loss_result_t* out);

cverl_status_t cverl_ppo_clipped_loss_backward_f32_cpu(
    cverl_const_tensor2d_t old_log_prob,
    cverl_const_tensor2d_t log_prob,
    cverl_const_tensor2d_t advantages,
    cverl_const_tensor2d_t response_mask,
    float clip_ratio,
    float clip_ratio_low,
    float clip_ratio_high,
    float clip_ratio_c,
    cverl_loss_agg_mode_t agg_mode,
    cverl_tensor2d_t grad_log_prob);

#ifdef __cplusplus
}
#endif
