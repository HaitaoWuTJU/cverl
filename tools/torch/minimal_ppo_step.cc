#include "cverl/torch/core_algos_torch.h"

#include <iostream>

namespace {

struct TinyPolicyImpl : torch::nn::Module {
  TinyPolicyImpl(int64_t obs_dim, int64_t hidden_dim, int64_t action_dim)
      : fc1(register_module("fc1", torch::nn::Linear(obs_dim, hidden_dim))),
        fc2(register_module("fc2", torch::nn::Linear(hidden_dim, action_dim))) {}

  torch::Tensor forward(const torch::Tensor& obs) {
    return fc2(torch::relu(fc1(obs)));
  }

  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};
};

TORCH_MODULE(TinyPolicy);

torch::Tensor action_log_probs(const torch::Tensor& logits, const torch::Tensor& actions) {
  torch::Tensor log_probs = torch::log_softmax(logits, /*dim=*/-1);
  return log_probs.gather(/*dim=*/-1, actions.unsqueeze(-1)).squeeze(-1);
}

}  // namespace

int main() {
  torch::manual_seed(11);

  constexpr int64_t batch = 8;
  constexpr int64_t seq = 4;
  constexpr int64_t obs_dim = 6;
  constexpr int64_t hidden_dim = 16;
  constexpr int64_t action_dim = 5;

  TinyPolicy policy(obs_dim, hidden_dim, action_dim);
  torch::optim::AdamW optimizer(policy->parameters(), torch::optim::AdamWOptions(1.0e-3));

  torch::Tensor obs = torch::randn({batch, seq, obs_dim});
  torch::Tensor actions = torch::randint(action_dim, {batch, seq}, torch::kLong);
  torch::Tensor response_mask = torch::ones({batch, seq}, torch::kFloat32);
  response_mask.index_put_({torch::indexing::Slice(), seq - 1}, torch::tensor(0.0f));

  torch::Tensor old_logits;
  {
    torch::NoGradGuard no_grad;
    old_logits = policy->forward(obs).detach();
  }
  torch::Tensor old_log_probs = action_log_probs(old_logits, actions).detach();

  torch::Tensor token_rewards = torch::randn({batch, seq}) * response_mask;
  torch::Tensor group_ids = torch::arange(batch, torch::kLong) / 2;
  torch::Tensor returns;
  torch::Tensor advantages = cverl::torch_backend::grpo_outcome_advantage(
      token_rewards, response_mask, group_ids, 1.0e-6, true, &returns);

  torch::Tensor logits = policy->forward(obs);
  torch::Tensor log_probs = action_log_probs(logits, actions);
  auto loss = cverl::torch_backend::ppo_clipped_loss(
      old_log_probs,
      log_probs,
      advantages,
      response_mask,
      0.2,
      -1.0,
      -1.0,
      3.0,
      CVERL_LOSS_AGG_TOKEN_MEAN);

  optimizer.zero_grad();
  loss.pg_loss.backward();
  optimizer.step();

  std::cout << "minimal PPO step completed\n";
  std::cout << "pg_loss=" << loss.pg_loss.item<float>() << "\n";
  std::cout << "ppo_kl=" << loss.ppo_kl.item<float>() << "\n";
  std::cout << "clipfrac=" << loss.pg_clipfrac.item<float>() << "\n";
  return 0;
}
