#pragma once

#include <torch/torch.h>

#include "cverl/core_algos.h"

namespace cverl::torch_backend {

struct PpoLossResult {
  torch::Tensor pg_loss;
  torch::Tensor pg_clipfrac;
  torch::Tensor ppo_kl;
  torch::Tensor pg_clipfrac_lower;
};

torch::Tensor masked_sum(const torch::Tensor& values, const torch::Tensor& mask);

torch::Tensor masked_mean(const torch::Tensor& values, const torch::Tensor& mask);

torch::Tensor masked_whiten(const torch::Tensor& values, const torch::Tensor& mask, bool shift_mean = true);

torch::Tensor kl_penalty(
    const torch::Tensor& logprob,
    const torch::Tensor& ref_logprob,
    cverl_kl_penalty_t penalty);

torch::Tensor gae_advantage_return(
    const torch::Tensor& token_level_rewards,
    const torch::Tensor& values,
    const torch::Tensor& response_mask,
    double gamma,
    double lam,
    torch::Tensor* returns);

torch::Tensor grpo_outcome_advantage(
    const torch::Tensor& token_level_rewards,
    const torch::Tensor& response_mask,
    const torch::Tensor& group_ids,
    double epsilon,
    bool norm_adv_by_std,
    torch::Tensor* returns);

PpoLossResult ppo_clipped_loss(
    const torch::Tensor& old_log_prob,
    const torch::Tensor& log_prob,
    const torch::Tensor& advantages,
    const torch::Tensor& response_mask,
    double clip_ratio,
    double clip_ratio_low,
    double clip_ratio_high,
    double clip_ratio_c,
    cverl_loss_agg_mode_t agg_mode);

}  // namespace cverl::torch_backend
