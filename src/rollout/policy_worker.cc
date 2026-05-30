#include "cverl/rollout/policy_worker.h"

#include <torch/torch.h>

#include <limits>
#include <stdexcept>
#include <utility>

#include "cverl/distributed/weight_sync.h"

namespace cverl::rollout {
namespace {

torch::Device policy_device(cverl::torch_backend::CausalLmPolicy& policy) {
  auto params = policy.parameters();
  if (!params.empty()) {
    return params.front().device();
  }
  return torch::Device(torch::kCPU);
}

torch::Tensor apply_top_k(torch::Tensor scores, int64_t top_k) {
  const int64_t vocab = scores.size(-1);
  if (top_k <= 0 || top_k >= vocab) {
    return scores;
  }
  auto top = scores.topk(top_k, /*dim=*/-1);
  auto values = std::get<0>(top);
  auto indices = std::get<1>(top);
  auto filtered = torch::full_like(scores, -std::numeric_limits<float>::infinity());
  return filtered.scatter(/*dim=*/-1, indices, values);
}

torch::Tensor sample_next_tokens(const torch::Tensor& logits, const GenerationConfig& config) {
  auto scores = logits.to(torch::kFloat32);
  if (config.temperature <= 0.0) {
    return scores.argmax(/*dim=*/-1);
  }
  scores = apply_top_k(scores / config.temperature, config.top_k);
  auto probs = torch::softmax(scores, /*dim=*/-1);

  if (config.top_p > 0.0 && config.top_p < 1.0) {
    auto sorted_pair = probs.sort(/*dim=*/-1, /*descending=*/true);
    auto sorted_probs = std::get<0>(sorted_pair);
    auto sorted_indices = std::get<1>(sorted_pair);
    auto keep = sorted_probs.cumsum(/*dim=*/-1) <= config.top_p;
    keep.index_put_({torch::indexing::Slice(), 0}, true);
    auto filtered = torch::where(keep, sorted_probs, torch::zeros_like(sorted_probs));
    filtered = filtered / filtered.sum(/*dim=*/-1, /*keepdim=*/true).clamp_min(1.0e-12);
    auto sampled_sorted = torch::multinomial(filtered, /*num_samples=*/1);
    return sorted_indices.gather(/*dim=*/1, sampled_sorted).squeeze(/*dim=*/1);
  }

  return torch::multinomial(probs, /*num_samples=*/1).squeeze(/*dim=*/1);
}

}  // namespace

PolicyRolloutWorker::PolicyRolloutWorker(std::shared_ptr<cverl::torch_backend::CausalLmPolicy> policy)
    : policy_(std::move(policy)) {
  if (!policy_) {
    throw std::invalid_argument("PolicyRolloutWorker requires a non-null policy");
  }
}

GenerationOutput PolicyRolloutWorker::generate(const TokenBatch& prompts,
                                               const GenerationConfig& config) {
  if (!prompts.token_ids.defined()) {
    throw std::invalid_argument("PolicyRolloutWorker prompts.token_ids is undefined");
  }
  if (prompts.token_ids.dim() != 2) {
    throw std::invalid_argument("PolicyRolloutWorker prompts.token_ids must be [B, T]");
  }
  if (config.max_new_tokens <= 0) {
    throw std::invalid_argument("PolicyRolloutWorker max_new_tokens must be positive");
  }

  torch::NoGradGuard no_grad;
  if (config.seed != 0) {
    torch::manual_seed(static_cast<int64_t>(config.seed));
  }

  const auto device = policy_device(*policy_);
  const int64_t batch = prompts.token_ids.size(0);
  const int64_t pad_id = policy_->pad_id();
  auto prompt_ids = prompts.token_ids.to(torch::TensorOptions().dtype(torch::kLong).device(device));

  std::vector<torch::Tensor> token_steps;
  std::vector<torch::Tensor> logprob_steps;
  token_steps.reserve(static_cast<size_t>(config.max_new_tokens));
  logprob_steps.reserve(static_cast<size_t>(config.max_new_tokens));

  auto lengths = torch::zeros({batch}, torch::TensorOptions().dtype(torch::kLong).device(device));
  auto finished = torch::zeros({batch}, torch::TensorOptions().dtype(torch::kBool).device(device));
  auto active = torch::logical_not(finished);

  for (int64_t step = 0; step < config.max_new_tokens; ++step) {
    torch::Tensor response_context;
    if (token_steps.empty()) {
      response_context = torch::full({batch, 1}, pad_id,
                                     torch::TensorOptions().dtype(torch::kLong).device(device));
    } else {
      response_context = torch::cat(token_steps, /*dim=*/1);
      auto pad_col = torch::full({batch, 1}, pad_id, response_context.options());
      response_context = torch::cat({response_context, pad_col}, /*dim=*/1);
    }

    auto logits = policy_->forward(prompt_ids, response_context).index({torch::indexing::Slice(), -1});
    auto next = sample_next_tokens(logits, config).to(torch::TensorOptions().dtype(torch::kLong).device(device));
    auto log_probs = torch::log_softmax(logits.to(torch::kFloat32), /*dim=*/-1)
                         .gather(/*dim=*/1, next.unsqueeze(1))
                         .squeeze(1);

    active = torch::logical_not(finished);
    auto emitted = torch::where(active, next, torch::full_like(next, pad_id));
    auto emitted_log_probs = torch::where(active, log_probs, torch::zeros_like(log_probs));
    token_steps.push_back(emitted.unsqueeze(1));
    logprob_steps.push_back(emitted_log_probs.unsqueeze(1));
    lengths = lengths + active.to(torch::kLong);

    if (config.eos_token_id >= 0) {
      finished = torch::logical_or(finished, torch::logical_and(active, next == config.eos_token_id));
      if (finished.all().item<bool>()) {
        break;
      }
    }
  }

  while (static_cast<int64_t>(token_steps.size()) < config.max_new_tokens) {
    token_steps.push_back(torch::full({batch, 1}, pad_id,
                                      torch::TensorOptions().dtype(torch::kLong).device(device)));
    logprob_steps.push_back(torch::zeros({batch, 1},
                                         torch::TensorOptions().dtype(torch::kFloat32).device(device)));
  }

  GenerationOutput out;
  out.token_ids = torch::cat(token_steps, /*dim=*/1).contiguous();
  out.logprobs = torch::cat(logprob_steps, /*dim=*/1).contiguous();
  out.lengths = lengths.contiguous();
  return out;
}

std::vector<cverl::distributed::ParameterView> PolicyRolloutWorker::actor_parameters() {
  return cverl::distributed::module_parameter_views(*policy_, /*recurse=*/true);
}

}  // namespace cverl::rollout
