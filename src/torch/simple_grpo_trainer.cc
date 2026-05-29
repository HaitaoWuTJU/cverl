#include "cverl/torch/simple_grpo_trainer.h"

#include <torch/torch.h>

#include <memory>
#include <stdexcept>
#include <string>

#include "cverl/torch/core_algos_torch.h"

namespace cverl::torch_backend {
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
  return torch::log_softmax(logits, -1).gather(-1, actions.unsqueeze(-1)).squeeze(-1);
}

torch::Tensor categorical_entropy(const torch::Tensor& logits) {
  torch::Tensor log_probs = torch::log_softmax(logits, -1);
  torch::Tensor probs = torch::softmax(logits, -1);
  return -(probs * log_probs).sum(-1);
}

}  // namespace

struct SimpleGrpoTrainer::Impl {
  explicit Impl(const SimpleGrpoTrainerConfig& cfg)
      : policy(cfg.action_dim, cfg.hidden_dim, cfg.action_dim),
        optimizer(policy->parameters(), torch::optim::AdamWOptions(cfg.learning_rate)) {}

  TinyPolicy policy;
  torch::optim::AdamW optimizer;
  int64_t step = 0;
};

SimpleGrpoTrainer::SimpleGrpoTrainer(SimpleGrpoTrainerConfig config) : config_(config) {
  if (config_.steps <= 0 || config_.prompts_per_batch <= 0 || config_.responses_per_prompt <= 1 ||
      config_.seq_len <= 0 || config_.action_dim <= 1 || config_.hidden_dim <= 0 || config_.ppo_epochs <= 0) {
    throw std::invalid_argument("invalid SimpleGrpoTrainerConfig");
  }
  torch::manual_seed(static_cast<int64_t>(config_.seed));
  impl_ = std::make_unique<Impl>(config_);
}

SimpleGrpoTrainer::~SimpleGrpoTrainer() = default;

SimpleGrpoTrainer::SimpleGrpoTrainer(SimpleGrpoTrainer&&) noexcept = default;

SimpleGrpoTrainer& SimpleGrpoTrainer::operator=(SimpleGrpoTrainer&&) noexcept = default;

SimpleGrpoTrainerMetrics SimpleGrpoTrainer::train_step() {
  const int64_t prompts = config_.prompts_per_batch;
  const int64_t group = config_.responses_per_prompt;
  const int64_t batch = prompts * group;
  const int64_t seq = config_.seq_len;
  const int64_t action_dim = config_.action_dim;

  torch::Tensor prompt_targets = torch::randint(action_dim, {prompts, seq}, torch::kLong);
  torch::Tensor targets = prompt_targets.repeat_interleave(group, 0);
  torch::Tensor obs = torch::one_hot(targets, action_dim).to(torch::kFloat32);
  torch::Tensor response_mask = torch::ones({batch, seq}, torch::kFloat32);
  torch::Tensor group_ids = torch::arange(prompts, torch::kLong).repeat_interleave(group);

  torch::Tensor old_logits;
  torch::Tensor actions;
  torch::Tensor old_log_probs;
  {
    torch::NoGradGuard no_grad;
    old_logits = impl_->policy->forward(obs);
    torch::Tensor probs = torch::softmax(old_logits, -1);
    actions = probs.reshape({batch * seq, action_dim}).multinomial(1).reshape({batch, seq});
    old_log_probs = action_log_probs(old_logits, actions).detach();
  }

  torch::Tensor token_correct = (actions == targets).to(torch::kFloat32) * response_mask;
  torch::Tensor scalar_rewards = token_correct.sum(-1) / response_mask.sum(-1);
  torch::Tensor token_rewards = torch::zeros({batch, seq}, torch::kFloat32);
  token_rewards.index_put_({torch::indexing::Slice(), seq - 1}, scalar_rewards);

  torch::Tensor returns;
  torch::Tensor advantages = grpo_outcome_advantage(
      token_rewards, response_mask, group_ids, 1.0e-6, true, &returns);

  SimpleGrpoTrainerMetrics last_metrics{};
  for (int64_t epoch = 0; epoch < config_.ppo_epochs; ++epoch) {
    torch::Tensor logits = impl_->policy->forward(obs);
    torch::Tensor log_probs = action_log_probs(logits, actions);
    auto loss_result = ppo_clipped_loss(
        old_log_probs,
        log_probs,
        advantages,
        response_mask,
        config_.clip_ratio,
        -1.0,
        -1.0,
        3.0,
        CVERL_LOSS_AGG_TOKEN_MEAN);

    torch::Tensor loss = loss_result.pg_loss;
    if (config_.entropy_coef != 0.0) {
      loss = loss - config_.entropy_coef * masked_mean(categorical_entropy(logits), response_mask);
    }

    impl_->optimizer.zero_grad();
    loss.backward();
    impl_->optimizer.step();

    last_metrics.loss = loss.item<double>();
    last_metrics.ppo_kl = loss_result.ppo_kl.item<double>();
    last_metrics.clipfrac = loss_result.pg_clipfrac.item<double>();
  }

  impl_->step += 1;
  last_metrics.step = impl_->step;
  last_metrics.avg_reward = scalar_rewards.mean().item<double>();
  last_metrics.success_rate = (scalar_rewards >= 0.999).to(torch::kFloat32).mean().item<double>();
  return last_metrics;
}

std::vector<SimpleGrpoTrainerMetrics> SimpleGrpoTrainer::train() {
  std::vector<SimpleGrpoTrainerMetrics> metrics;
  metrics.reserve(static_cast<size_t>(config_.steps));
  for (int64_t i = 0; i < config_.steps; ++i) {
    metrics.push_back(train_step());
  }
  return metrics;
}

void SimpleGrpoTrainer::save_checkpoint(const std::string& prefix) const {
  torch::serialize::OutputArchive model_archive;
  impl_->policy->save(model_archive);
  model_archive.save_to(prefix + ".model.pt");

  torch::serialize::OutputArchive optim_archive;
  impl_->optimizer.save(optim_archive);
  optim_archive.save_to(prefix + ".optim.pt");

  torch::serialize::OutputArchive meta_archive;
  meta_archive.write("step", torch::tensor(impl_->step, torch::kInt64));
  meta_archive.write("seed", torch::tensor(static_cast<int64_t>(config_.seed), torch::kInt64));
  meta_archive.save_to(prefix + ".meta.pt");
}

void SimpleGrpoTrainer::load_checkpoint(const std::string& prefix) {
  torch::serialize::InputArchive model_archive;
  model_archive.load_from(prefix + ".model.pt");
  impl_->policy->load(model_archive);

  torch::serialize::InputArchive optim_archive;
  optim_archive.load_from(prefix + ".optim.pt");
  impl_->optimizer.load(optim_archive);

  torch::serialize::InputArchive meta_archive;
  meta_archive.load_from(prefix + ".meta.pt");
  torch::Tensor step;
  meta_archive.read("step", step);
  impl_->step = step.item<int64_t>();
}

int64_t SimpleGrpoTrainer::current_step() const {
  return impl_->step;
}

}  // namespace cverl::torch_backend
