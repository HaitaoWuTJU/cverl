#include "cverl/torch/core_algos_torch.h"

#include <limits>
#include <unordered_map>
#include <vector>

namespace cverl::torch_backend {
namespace {

constexpr double kEps = 1.0e-8;

torch::Tensor aggregate_loss(
    const torch::Tensor& loss_mat,
    const torch::Tensor& loss_mask,
    cverl_loss_agg_mode_t mode) {
  if (mode == CVERL_LOSS_AGG_TOKEN_MEAN) {
    return masked_sum(loss_mat, loss_mask) / (loss_mask.sum() + kEps);
  }

  torch::Tensor seq_token_count = loss_mask.sum(/*dim=*/-1);
  torch::Tensor seq_mask = (seq_token_count > 0).to(loss_mat.dtype());
  torch::Tensor seq_losses = (loss_mat * loss_mask).sum(/*dim=*/-1);
  if (mode == CVERL_LOSS_AGG_SEQ_MEAN_TOKEN_MEAN) {
    seq_losses = seq_losses / (seq_token_count + kEps);
  }
  if (mode == CVERL_LOSS_AGG_SEQ_MEAN_TOKEN_SUM || mode == CVERL_LOSS_AGG_SEQ_MEAN_TOKEN_MEAN) {
    return masked_sum(seq_losses, seq_mask) / (seq_mask.sum() + kEps);
  }
  TORCH_CHECK(false, "unsupported loss aggregation mode");
}

}  // namespace

torch::Tensor masked_sum(const torch::Tensor& values, const torch::Tensor& mask) {
  return torch::where(mask.to(torch::kBool), values, torch::zeros_like(values)).mul(mask).sum();
}

torch::Tensor masked_mean(const torch::Tensor& values, const torch::Tensor& mask) {
  return masked_sum(values, mask) / (mask.sum() + kEps);
}

torch::Tensor masked_whiten(const torch::Tensor& values, const torch::Tensor& mask, bool shift_mean) {
  torch::Tensor mean = masked_mean(values, mask);
  torch::Tensor centered = values - mean;
  torch::Tensor var = masked_mean(centered.pow(2), mask);
  torch::Tensor mask_sum = mask.sum();
  var = var * mask_sum / (mask_sum - 1.0);
  torch::Tensor whitened = centered * torch::rsqrt(var + kEps);
  if (!shift_mean) {
    whitened = whitened + mean;
  }
  return whitened;
}

torch::Tensor kl_penalty(
    const torch::Tensor& logprob,
    const torch::Tensor& ref_logprob,
    cverl_kl_penalty_t penalty) {
  torch::Tensor diff = logprob - ref_logprob;
  switch (penalty) {
    case CVERL_KL_K1:
      return diff;
    case CVERL_KL_ABS:
      return diff.abs();
    case CVERL_KL_K2:
      return 0.5 * diff.square();
    case CVERL_KL_K3: {
      torch::Tensor kl = torch::clamp(ref_logprob - logprob, -20.0, 20.0);
      return torch::clamp(torch::exp(kl) - kl - 1.0, -10.0, 10.0);
    }
  }
  TORCH_CHECK(false, "unsupported KL penalty");
}

torch::Tensor gae_advantage_return(
    const torch::Tensor& token_level_rewards,
    const torch::Tensor& values,
    const torch::Tensor& response_mask,
    double gamma,
    double lam,
    torch::Tensor* returns) {
  TORCH_CHECK(token_level_rewards.dim() == 2, "GAE expects [batch, seq] rewards");
  TORCH_CHECK(values.sizes() == token_level_rewards.sizes(), "values shape mismatch");
  TORCH_CHECK(response_mask.sizes() == token_level_rewards.sizes(), "response_mask shape mismatch");

  torch::NoGradGuard no_grad;
  int64_t rows = token_level_rewards.size(0);
  int64_t cols = token_level_rewards.size(1);
  torch::Tensor advantages = torch::empty_like(token_level_rewards);

  for (int64_t r = 0; r < rows; ++r) {
    torch::Tensor next_value = torch::zeros({}, token_level_rewards.options());
    torch::Tensor last_gae_lam = torch::zeros({}, token_level_rewards.options());
    for (int64_t c = cols - 1; c >= 0; --c) {
      torch::Tensor reward = token_level_rewards.index({r, c});
      torch::Tensor value = values.index({r, c});
      torch::Tensor mask = response_mask.index({r, c});
      torch::Tensor delta = reward + gamma * next_value - value;
      torch::Tensor candidate = delta + gamma * lam * last_gae_lam;
      next_value = value * mask + (1.0 - mask) * next_value;
      last_gae_lam = candidate * mask + (1.0 - mask) * last_gae_lam;
      advantages.index_put_({r, c}, last_gae_lam);
    }
  }

  if (returns != nullptr) {
    *returns = advantages + values;
  }
  return masked_whiten(advantages, response_mask);
}

torch::Tensor grpo_outcome_advantage(
    const torch::Tensor& token_level_rewards,
    const torch::Tensor& response_mask,
    const torch::Tensor& group_ids,
    double epsilon,
    bool norm_adv_by_std,
    torch::Tensor* returns) {
  TORCH_CHECK(token_level_rewards.dim() == 2, "GRPO expects [batch, seq] rewards");
  TORCH_CHECK(response_mask.sizes() == token_level_rewards.sizes(), "response_mask shape mismatch");
  TORCH_CHECK(group_ids.dim() == 1 && group_ids.size(0) == token_level_rewards.size(0), "group_ids shape mismatch");

  torch::NoGradGuard no_grad;
  torch::Tensor scores = token_level_rewards.sum(/*dim=*/-1);
  torch::Tensor group_cpu = group_ids.to(torch::kCPU).to(torch::kLong).contiguous();
  auto group_acc = group_cpu.accessor<int64_t, 1>();

  std::unordered_map<int64_t, std::vector<int64_t>> groups;
  for (int64_t i = 0; i < group_ids.size(0); ++i) {
    groups[group_acc[i]].push_back(i);
  }

  torch::Tensor scalars = torch::empty_like(scores);
  for (const auto& entry : groups) {
    std::vector<torch::Tensor> group_scores;
    group_scores.reserve(entry.second.size());
    for (int64_t row : entry.second) {
      group_scores.push_back(scores.index({row}));
    }
    torch::Tensor stacked = torch::stack(group_scores);
    torch::Tensor mean = entry.second.size() == 1 ? torch::zeros({}, scores.options()) : stacked.mean();
    torch::Tensor std = entry.second.size() == 1 ? torch::ones({}, scores.options()) : stacked.std(/*unbiased=*/true);
    for (int64_t row : entry.second) {
      torch::Tensor scalar = scores.index({row}) - mean;
      if (norm_adv_by_std) {
        scalar = scalar / (std + epsilon);
      }
      scalars.index_put_({row}, scalar);
    }
  }

  torch::Tensor advantages = scalars.unsqueeze(-1) * response_mask;
  if (returns != nullptr) {
    *returns = advantages;
  }
  return advantages;
}

PpoLossResult ppo_clipped_loss(
    const torch::Tensor& old_log_prob,
    const torch::Tensor& log_prob,
    const torch::Tensor& advantages,
    const torch::Tensor& response_mask,
    double clip_ratio,
    double clip_ratio_low,
    double clip_ratio_high,
    double clip_ratio_c,
    cverl_loss_agg_mode_t agg_mode) {
  TORCH_CHECK(clip_ratio_c > 1.0, "clip_ratio_c must be > 1.0");
  if (clip_ratio_low < 0.0) {
    clip_ratio_low = clip_ratio;
  }
  if (clip_ratio_high < 0.0) {
    clip_ratio_high = clip_ratio;
  }

  torch::Tensor negative_approx_kl = torch::clamp(log_prob - old_log_prob, -20.0, 20.0);
  torch::Tensor ratio = torch::exp(negative_approx_kl);
  torch::Tensor ppo_kl = masked_mean(-negative_approx_kl, response_mask);

  torch::Tensor pg_losses1 = -advantages * ratio;
  torch::Tensor pg_losses2 = -advantages * torch::clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high);
  torch::Tensor clip_pg_losses1 = torch::maximum(pg_losses1, pg_losses2);
  torch::Tensor pg_clipfrac = masked_mean((pg_losses2 > pg_losses1).to(response_mask.dtype()), response_mask);

  torch::Tensor pg_losses3 = -advantages * clip_ratio_c;
  torch::Tensor clip_pg_losses2 = torch::minimum(pg_losses3, clip_pg_losses1);
  torch::Tensor pg_clipfrac_lower =
      masked_mean(((clip_pg_losses1 > pg_losses3) * (advantages < 0)).to(response_mask.dtype()), response_mask);
  torch::Tensor pg_losses = torch::where(advantages < 0, clip_pg_losses2, clip_pg_losses1);

  return PpoLossResult{
      aggregate_loss(pg_losses, response_mask, agg_mode),
      pg_clipfrac,
      ppo_kl,
      pg_clipfrac_lower,
  };
}

}  // namespace cverl::torch_backend
