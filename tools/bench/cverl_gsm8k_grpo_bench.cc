#include "cverl/data/hf_dataset.h"
#include "cverl/torch/core_algos_torch.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace {

struct TinyPolicyImpl : torch::nn::Module {
  TinyPolicyImpl(int64_t action_dim, int64_t hidden_dim)
      : fc1(register_module("fc1", torch::nn::Linear(action_dim, hidden_dim))),
        fc2(register_module("fc2", torch::nn::Linear(hidden_dim, action_dim))) {}

  torch::Tensor forward(const torch::Tensor& obs) {
    return fc2(torch::relu(fc1(obs)));
  }

  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};
};

TORCH_MODULE(TinyPolicy);

struct Args {
  std::string dataset;
  int64_t steps = 32;
  int64_t prompts = 8;
  int64_t responses = 4;
  int64_t seq_len = 16;
  int64_t action_dim = 64;
  int64_t hidden_dim = 128;
  int64_t ppo_epochs = 2;
  double lr = 3.0e-3;
  uint64_t seed = 17;
  std::string device = "cpu";
};

const char* require_value(int& i, int argc, char** argv, const char* name) {
  if (i + 1 >= argc) {
    throw std::invalid_argument(std::string(name) + " requires a value");
  }
  return argv[++i];
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--dataset") {
      args.dataset = require_value(i, argc, argv, "--dataset");
    } else if (arg == "--steps") {
      args.steps = std::strtoll(require_value(i, argc, argv, "--steps"), nullptr, 10);
    } else if (arg == "--prompts") {
      args.prompts = std::strtoll(require_value(i, argc, argv, "--prompts"), nullptr, 10);
    } else if (arg == "--responses") {
      args.responses = std::strtoll(require_value(i, argc, argv, "--responses"), nullptr, 10);
    } else if (arg == "--seq-len") {
      args.seq_len = std::strtoll(require_value(i, argc, argv, "--seq-len"), nullptr, 10);
    } else if (arg == "--action-dim") {
      args.action_dim = std::strtoll(require_value(i, argc, argv, "--action-dim"), nullptr, 10);
    } else if (arg == "--hidden-dim") {
      args.hidden_dim = std::strtoll(require_value(i, argc, argv, "--hidden-dim"), nullptr, 10);
    } else if (arg == "--ppo-epochs") {
      args.ppo_epochs = std::strtoll(require_value(i, argc, argv, "--ppo-epochs"), nullptr, 10);
    } else if (arg == "--lr") {
      args.lr = std::strtod(require_value(i, argc, argv, "--lr"), nullptr);
    } else if (arg == "--seed") {
      args.seed = static_cast<uint64_t>(std::strtoull(require_value(i, argc, argv, "--seed"), nullptr, 10));
    } else if (arg == "--device") {
      args.device = require_value(i, argc, argv, "--device");
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  if (args.dataset.empty()) {
    throw std::invalid_argument("--dataset is required");
  }
  if (args.steps <= 0 || args.prompts <= 0 || args.responses <= 1 || args.seq_len <= 0 ||
      args.action_dim <= 1 || args.hidden_dim <= 0 || args.ppo_epochs <= 0) {
    throw std::invalid_argument("invalid benchmark shape");
  }
  return args;
}

uint64_t fnv1a64(const std::string& text, uint64_t seed) {
  uint64_t hash = 1469598103934665603ULL ^ seed;
  for (unsigned char ch : text) {
    hash ^= ch;
    hash *= 1099511628211ULL;
  }
  return hash;
}

torch::Device parse_device(const std::string& value) {
  if (value == "cpu") {
    return torch::Device(torch::kCPU);
  }
  if (value.rfind("cuda", 0) == 0) {
    size_t colon = value.find(':');
    int index = colon == std::string::npos ? 0 : std::stoi(value.substr(colon + 1));
    return torch::Device(torch::kCUDA, index);
  }
  throw std::invalid_argument("unsupported device: " + value);
}

torch::Tensor make_targets(const std::vector<cverl::data::PromptAnswerExample>& data,
                           int64_t step,
                           const Args& args,
                           torch::Device device) {
  std::vector<int64_t> values(static_cast<size_t>(args.prompts * args.seq_len));
  for (int64_t row = 0; row < args.prompts; ++row) {
    const auto& example = data[static_cast<size_t>((step * args.prompts + row) % static_cast<int64_t>(data.size()))];
    uint64_t state = fnv1a64(example.prompt + "\n" + example.answer, args.seed + static_cast<uint64_t>(step));
    for (int64_t col = 0; col < args.seq_len; ++col) {
      state ^= state >> 12;
      state ^= state << 25;
      state ^= state >> 27;
      values[static_cast<size_t>(row * args.seq_len + col)] =
          static_cast<int64_t>((state * 2685821657736338717ULL) % static_cast<uint64_t>(args.action_dim));
    }
  }
  return torch::from_blob(values.data(), {args.prompts, args.seq_len}, torch::kInt64).clone().to(device);
}

torch::Tensor action_log_probs(const torch::Tensor& logits, const torch::Tensor& actions) {
  return torch::log_softmax(logits, -1).gather(-1, actions.unsqueeze(-1)).squeeze(-1);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);
    torch::manual_seed(static_cast<int64_t>(args.seed));
    torch::Device device = parse_device(args.device);

    cverl::data::JsonlDatasetOptions dataset_options;
    dataset_options.path = args.dataset;
    auto data = cverl::data::load_prompt_answer_jsonl(dataset_options);
    if (data.empty()) {
      throw std::runtime_error("dataset is empty");
    }

    TinyPolicy policy(args.action_dim, args.hidden_dim);
    policy->to(device);
    torch::optim::AdamW optimizer(policy->parameters(), torch::optim::AdamWOptions(args.lr));

    const int64_t batch = args.prompts * args.responses;
    const int64_t total_sequences = args.steps * batch;
    const int64_t total_tokens = total_sequences * args.seq_len * args.ppo_epochs;

    double loss_value = 0.0;
    double reward_value = 0.0;
    double kl_value = 0.0;
    double clipfrac_value = 0.0;

    auto start = std::chrono::steady_clock::now();
    for (int64_t step = 0; step < args.steps; ++step) {
      auto prompt_targets = make_targets(data, step, args, device);
      auto targets = prompt_targets.repeat_interleave(args.responses, 0);
      auto obs = torch::one_hot(targets, args.action_dim).to(torch::kFloat32).to(device);
      auto response_mask = torch::ones({batch, args.seq_len}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
      auto group_ids = torch::arange(args.prompts, torch::TensorOptions().device(device).dtype(torch::kInt64))
                           .repeat_interleave(args.responses);

      torch::Tensor actions;
      torch::Tensor old_log_probs;
      {
        torch::NoGradGuard no_grad;
        auto old_logits = policy->forward(obs);
        auto probs = torch::softmax(old_logits, -1);
        actions = probs.reshape({batch * args.seq_len, args.action_dim}).multinomial(1).reshape({batch, args.seq_len});
        old_log_probs = action_log_probs(old_logits, actions).detach();
      }

      auto token_correct = (actions == targets).to(torch::kFloat32) * response_mask;
      auto scalar_rewards = token_correct.sum(-1) / response_mask.sum(-1);
      auto token_rewards = torch::zeros({batch, args.seq_len}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
      token_rewards.index_put_({torch::indexing::Slice(), args.seq_len - 1}, scalar_rewards);

      torch::Tensor returns;
      auto advantages = cverl::torch_backend::grpo_outcome_advantage(
          token_rewards, response_mask, group_ids, 1.0e-6, true, &returns);

      for (int64_t epoch = 0; epoch < args.ppo_epochs; ++epoch) {
        auto logits = policy->forward(obs);
        auto log_probs = action_log_probs(logits, actions);
        auto loss = cverl::torch_backend::ppo_clipped_loss(
            old_log_probs, log_probs, advantages, response_mask, 0.2, -1.0, -1.0, 3.0, CVERL_LOSS_AGG_TOKEN_MEAN);
        optimizer.zero_grad();
        loss.pg_loss.backward();
        optimizer.step();

        loss_value = loss.pg_loss.item<double>();
        kl_value = loss.ppo_kl.item<double>();
        clipfrac_value = loss.pg_clipfrac.item<double>();
      }
      reward_value = scalar_rewards.mean().item<double>();
    }
    auto end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "backend,cverl\n";
    std::cout << "dataset_rows," << data.size() << "\n";
    std::cout << "steps," << args.steps << "\n";
    std::cout << "prompts," << args.prompts << "\n";
    std::cout << "responses," << args.responses << "\n";
    std::cout << "seq_len," << args.seq_len << "\n";
    std::cout << "action_dim," << args.action_dim << "\n";
    std::cout << "hidden_dim," << args.hidden_dim << "\n";
    std::cout << "ppo_epochs," << args.ppo_epochs << "\n";
    std::cout << "device," << args.device << "\n";
    std::cout << "seconds," << seconds << "\n";
    std::cout << "sequences_per_second," << (static_cast<double>(total_sequences) / seconds) << "\n";
    std::cout << "train_tokens_per_second," << (static_cast<double>(total_tokens) / seconds) << "\n";
    std::cout << "last_loss," << loss_value << "\n";
    std::cout << "last_avg_reward," << reward_value << "\n";
    std::cout << "last_ppo_kl," << kl_value << "\n";
    std::cout << "last_clipfrac," << clipfrac_value << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "cverl_gsm8k_grpo_bench failed: " << e.what() << "\n";
    return 1;
  }
}
