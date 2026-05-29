#include "cverl/model/hf_model_loader.h"
#include "cverl/torch/core_algos_torch.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

struct EmbeddingPolicyImpl : torch::nn::Module {
  explicit EmbeddingPolicyImpl(const torch::Tensor& embedding_subset)
      : frozen_embedding(register_buffer("frozen_embedding", embedding_subset.to(torch::kFloat32).contiguous())),
        adapter(register_module(
            "adapter",
            torch::nn::Linear(torch::nn::LinearOptions(embedding_subset.size(1), embedding_subset.size(1)).bias(false)))),
        lm_head(register_module(
            "lm_head",
            torch::nn::Linear(torch::nn::LinearOptions(embedding_subset.size(1), embedding_subset.size(0)).bias(true)))) {
    torch::NoGradGuard no_grad;
    adapter->weight.copy_(torch::eye(embedding_subset.size(1), torch::kFloat32));
    lm_head->weight.copy_(frozen_embedding);
    lm_head->bias.zero_();
  }

  torch::Tensor encode_targets(const torch::Tensor& token_ids) {
    return frozen_embedding.index_select(0, token_ids.reshape(-1)).reshape({token_ids.size(0), token_ids.size(1), -1});
  }

  torch::Tensor forward_from_targets(const torch::Tensor& token_ids) {
    torch::Tensor hidden = adapter(encode_targets(token_ids));
    return lm_head(torch::silu(hidden)) / std::sqrt(static_cast<double>(hidden.size(-1)));
  }

  torch::Tensor frozen_embedding;
  torch::nn::Linear adapter{nullptr};
  torch::nn::Linear lm_head{nullptr};
};

TORCH_MODULE(EmbeddingPolicy);

torch::Tensor action_log_probs(const torch::Tensor& logits, const torch::Tensor& actions) {
  return torch::log_softmax(logits, -1).gather(-1, actions.unsqueeze(-1)).squeeze(-1);
}

struct Args {
  std::string model_dir;
  int64_t steps = 8;
  int64_t prompts = 4;
  int64_t responses_per_prompt = 4;
  int64_t seq_len = 8;
  int64_t vocab_subset = 64;
  int64_t ppo_epochs = 2;
  double lr = 1.0e-3;
};

Args parse_args(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0]
              << " <model-dir> [--steps N] [--prompts N] [--responses N] [--seq-len N]"
                 " [--vocab-subset N] [--ppo-epochs N] [--lr LR]\n";
    std::exit(2);
  }
  Args args;
  args.model_dir = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    auto require_value = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << name << " requires a value\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--steps") {
      args.steps = std::strtoll(require_value("--steps"), nullptr, 10);
    } else if (arg == "--prompts") {
      args.prompts = std::strtoll(require_value("--prompts"), nullptr, 10);
    } else if (arg == "--responses") {
      args.responses_per_prompt = std::strtoll(require_value("--responses"), nullptr, 10);
    } else if (arg == "--seq-len") {
      args.seq_len = std::strtoll(require_value("--seq-len"), nullptr, 10);
    } else if (arg == "--vocab-subset") {
      args.vocab_subset = std::strtoll(require_value("--vocab-subset"), nullptr, 10);
    } else if (arg == "--ppo-epochs") {
      args.ppo_epochs = std::strtoll(require_value("--ppo-epochs"), nullptr, 10);
    } else if (arg == "--lr") {
      args.lr = std::strtod(require_value("--lr"), nullptr);
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      std::exit(2);
    }
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  torch::manual_seed(41);

  cverl::HfModelLoader loader(args.model_dir);
  loader.load_metadata();
  torch::Tensor embedding = loader.load_tensor("model.language_model.embed_tokens.weight").to(torch::kFloat32);
  if (args.vocab_subset <= 1 || args.vocab_subset > embedding.size(0)) {
    std::cerr << "invalid vocab subset size\n";
    return 2;
  }
  torch::Tensor embedding_subset = embedding.narrow(0, 0, args.vocab_subset).contiguous();

  EmbeddingPolicy policy(embedding_subset);
  torch::optim::AdamW optimizer(policy->parameters(), torch::optim::AdamWOptions(args.lr));

  std::cout << "loaded_model=" << args.model_dir << "\n";
  std::cout << "model_type=" << loader.config().model_type << "\n";
  std::cout << "embedding_subset=" << embedding_subset.sizes() << "\n";
  std::cout << "step,loss,avg_reward,success_rate,ppo_kl,clipfrac\n";

  const int64_t batch = args.prompts * args.responses_per_prompt;
  for (int64_t step = 1; step <= args.steps; ++step) {
    torch::Tensor prompt_targets = torch::randint(args.vocab_subset, {args.prompts, args.seq_len}, torch::kLong);
    torch::Tensor targets = prompt_targets.repeat_interleave(args.responses_per_prompt, 0);
    torch::Tensor response_mask = torch::ones({batch, args.seq_len}, torch::kFloat32);
    torch::Tensor group_ids = torch::arange(args.prompts, torch::kLong).repeat_interleave(args.responses_per_prompt);

    torch::Tensor actions;
    torch::Tensor old_log_probs;
    {
      torch::NoGradGuard no_grad;
      torch::Tensor old_logits = policy->forward_from_targets(targets);
      torch::Tensor probs = torch::softmax(old_logits, -1);
      actions = probs.reshape({batch * args.seq_len, args.vocab_subset}).multinomial(1).reshape({batch, args.seq_len});
      old_log_probs = action_log_probs(old_logits, actions).detach();
    }

    torch::Tensor token_correct = (actions == targets).to(torch::kFloat32) * response_mask;
    torch::Tensor scalar_rewards = token_correct.sum(-1) / response_mask.sum(-1);
    torch::Tensor token_rewards = torch::zeros({batch, args.seq_len}, torch::kFloat32);
    token_rewards.index_put_({torch::indexing::Slice(), args.seq_len - 1}, scalar_rewards);

    torch::Tensor returns;
    torch::Tensor advantages =
        cverl::torch_backend::grpo_outcome_advantage(token_rewards, response_mask, group_ids, 1.0e-6, true, &returns);

    double loss_value = 0.0;
    double kl_value = 0.0;
    double clipfrac_value = 0.0;
    for (int64_t epoch = 0; epoch < args.ppo_epochs; ++epoch) {
      torch::Tensor logits = policy->forward_from_targets(targets);
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

      loss_value = loss.pg_loss.item<double>();
      kl_value = loss.ppo_kl.item<double>();
      clipfrac_value = loss.pg_clipfrac.item<double>();
    }

    double avg_reward = scalar_rewards.mean().item<double>();
    double success = (scalar_rewards >= 0.999).to(torch::kFloat32).mean().item<double>();
    std::cout << step << "," << loss_value << "," << avg_reward << "," << success << "," << kl_value << ","
              << clipfrac_value << "\n";
  }

  return 0;
}
