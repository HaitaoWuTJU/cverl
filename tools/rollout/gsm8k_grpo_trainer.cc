// gsm8k_grpo_trainer
//
// End-to-end rollout-driven GRPO/PPO trainer:
//   GSM8K JSONL  ->  in-process actor rollout
//                 ->  GSM8K rule reward
//                 ->  Tokenizer
//                 ->  CausalLmPolicy logprobs
//                 ->  GRPO advantages -> PPO clipped step
//
// The rollout and trainer share the same actor parameters in-process. This is
// the baseline efficient path: no HTTP, no JSON transport, no checkpoint reload.
#include "cverl/data/hf_dataset.h"
#include "cverl/reward/gsm8k.h"
#include "cverl/rollout/rollout_batch.h"
#include "cverl/text/byte_tokenizer.h"
#include "cverl/text/hf_bpe_tokenizer.h"
#include "cverl/text/tokenizer.h"
#include "cverl/torch/causal_lm_policy.h"
#include "cverl/torch/core_algos_torch.h"
#include "cverl/torch/tiny_causal_policy.h"

#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string dataset;
  int64_t prompts_per_batch = 4;
  uint32_t n = 4;
  int64_t steps = 4;
  int64_t ppo_epochs = 2;
  int64_t max_prompt_tokens = 128;
  int64_t max_response_tokens = 128;
  int64_t hidden_dim = 32;
  uint32_t max_tokens = 128;
  double temperature = 1.0;
  double top_p = 1.0;
  double clip_ratio = 0.2;
  double learning_rate = 3.0e-3;
  double weight_decay = 0.0;
  double kl_coef = 0.0;
  cverl_kl_penalty_t kl_penalty_mode = CVERL_KL_K1;
  uint64_t seed = 17;
  cverl::reward::Gsm8kExtractionMethod reward_method = cverl::reward::Gsm8kExtractionMethod::Strict;
  std::string tokenizer_kind = "byte";
  std::string tokenizer_path;
  std::string policy_kind = "tiny";
  std::string model_dir;
  int64_t qwen_max_layers = -1;
  std::string device = "cpu";
  std::string export_dir;
  int64_t export_every = 0;
  std::string export_dtype = "bfloat16";
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
    std::string a = argv[i];
    if (a == "--dataset") {
      args.dataset = require_value(i, argc, argv, "--dataset");
    } else if (a == "--prompts") {
      args.prompts_per_batch = std::strtoll(require_value(i, argc, argv, "--prompts"), nullptr, 10);
    } else if (a == "--n") {
      args.n = static_cast<uint32_t>(std::strtoul(require_value(i, argc, argv, "--n"), nullptr, 10));
    } else if (a == "--steps") {
      args.steps = std::strtoll(require_value(i, argc, argv, "--steps"), nullptr, 10);
    } else if (a == "--ppo-epochs") {
      args.ppo_epochs = std::strtoll(require_value(i, argc, argv, "--ppo-epochs"), nullptr, 10);
    } else if (a == "--max-prompt-tokens") {
      args.max_prompt_tokens = std::strtoll(require_value(i, argc, argv, "--max-prompt-tokens"), nullptr, 10);
    } else if (a == "--max-response-tokens") {
      args.max_response_tokens = std::strtoll(require_value(i, argc, argv, "--max-response-tokens"), nullptr, 10);
    } else if (a == "--hidden-dim") {
      args.hidden_dim = std::strtoll(require_value(i, argc, argv, "--hidden-dim"), nullptr, 10);
    } else if (a == "--max-tokens") {
      args.max_tokens = static_cast<uint32_t>(std::strtoul(require_value(i, argc, argv, "--max-tokens"), nullptr, 10));
    } else if (a == "--temperature") {
      args.temperature = std::strtod(require_value(i, argc, argv, "--temperature"), nullptr);
    } else if (a == "--top-p") {
      args.top_p = std::strtod(require_value(i, argc, argv, "--top-p"), nullptr);
    } else if (a == "--clip-ratio") {
      args.clip_ratio = std::strtod(require_value(i, argc, argv, "--clip-ratio"), nullptr);
    } else if (a == "--lr") {
      args.learning_rate = std::strtod(require_value(i, argc, argv, "--lr"), nullptr);
    } else if (a == "--weight-decay") {
      args.weight_decay = std::strtod(require_value(i, argc, argv, "--weight-decay"), nullptr);
    } else if (a == "--kl-coef") {
      args.kl_coef = std::strtod(require_value(i, argc, argv, "--kl-coef"), nullptr);
    } else if (a == "--kl-penalty") {
      std::string m = require_value(i, argc, argv, "--kl-penalty");
      if (m == "k1") {
        args.kl_penalty_mode = CVERL_KL_K1;
      } else if (m == "abs") {
        args.kl_penalty_mode = CVERL_KL_ABS;
      } else if (m == "k2") {
        args.kl_penalty_mode = CVERL_KL_K2;
      } else if (m == "k3") {
        args.kl_penalty_mode = CVERL_KL_K3;
      } else {
        throw std::invalid_argument("--kl-penalty must be k1|abs|k2|k3");
      }
    } else if (a == "--seed") {
      args.seed = static_cast<uint64_t>(std::strtoull(require_value(i, argc, argv, "--seed"), nullptr, 10));
    } else if (a == "--reward-method") {
      std::string m = require_value(i, argc, argv, "--reward-method");
      if (m == "strict") {
        args.reward_method = cverl::reward::Gsm8kExtractionMethod::Strict;
      } else if (m == "flexible") {
        args.reward_method = cverl::reward::Gsm8kExtractionMethod::Flexible;
      } else {
        throw std::invalid_argument("--reward-method must be strict|flexible");
      }
    } else if (a == "--tokenizer") {
      args.tokenizer_kind = require_value(i, argc, argv, "--tokenizer");
    } else if (a == "--tokenizer-path") {
      args.tokenizer_path = require_value(i, argc, argv, "--tokenizer-path");
    } else if (a == "--policy") {
      args.policy_kind = require_value(i, argc, argv, "--policy");
    } else if (a == "--model-dir") {
      args.model_dir = require_value(i, argc, argv, "--model-dir");
    } else if (a == "--qwen-max-layers") {
      args.qwen_max_layers = std::strtoll(require_value(i, argc, argv, "--qwen-max-layers"), nullptr, 10);
    } else if (a == "--device") {
      args.device = require_value(i, argc, argv, "--device");
    } else if (a == "--export-dir") {
      args.export_dir = require_value(i, argc, argv, "--export-dir");
    } else if (a == "--export-every") {
      args.export_every = std::strtoll(require_value(i, argc, argv, "--export-every"), nullptr, 10);
    } else if (a == "--export-dtype") {
      args.export_dtype = require_value(i, argc, argv, "--export-dtype");
    } else {
      throw std::invalid_argument("unknown argument: " + a);
    }
  }
  if (args.dataset.empty()) {
    throw std::invalid_argument("--dataset is required");
  }
  if (args.prompts_per_batch <= 0 || args.n == 0 || args.steps <= 0 || args.ppo_epochs <= 0) {
    throw std::invalid_argument("invalid batch shape");
  }
  if (args.policy_kind != "tiny" && args.policy_kind != "qwen") {
    throw std::invalid_argument("--policy must be tiny|qwen");
  }
  if (args.policy_kind == "qwen" && args.model_dir.empty()) {
    throw std::invalid_argument("--policy qwen requires --model-dir");
  }
  if (!args.export_dir.empty() && args.export_every == 0) {
    args.export_every = 1;
  }
  if (args.export_every < 0) {
    throw std::invalid_argument("--export-every must be >= 0");
  }
  return args;
}

std::string step_export_path(const std::string& export_dir, int64_t step) {
  std::ostringstream name;
  name << "step_" << std::setw(6) << std::setfill('0') << step;
  return (std::filesystem::path(export_dir) / name.str()).string();
}

torch::Tensor ids_to_tensor(const std::vector<int32_t>& ids, torch::Device device) {
  auto cpu = torch::empty({1, static_cast<int64_t>(ids.size())},
                          torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  auto* ptr = cpu.data_ptr<int64_t>();
  for (size_t i = 0; i < ids.size(); ++i) {
    ptr[i] = static_cast<int64_t>(ids[i]);
  }
  if (device.is_cpu()) {
    return cpu;
  }
  return cpu.to(device, /*non_blocking=*/true);
}

int64_t sample_next_token(const torch::Tensor& logits, double temperature, double top_p) {
  auto scores = logits.to(torch::kFloat32);
  if (temperature <= 0.0) {
    return scores.argmax(-1).item<int64_t>();
  }
  scores = scores / temperature;
  auto probs = torch::softmax(scores, -1);
  if (top_p > 0.0 && top_p < 1.0) {
    auto sorted_pair = probs.sort(/*dim=*/-1, /*descending=*/true);
    auto sorted_probs = std::get<0>(sorted_pair);
    auto sorted_indices = std::get<1>(sorted_pair);
    auto keep = sorted_probs.cumsum(-1) <= top_p;
    keep.index_put_({0}, true);
    auto filtered = torch::where(keep, sorted_probs, torch::zeros_like(sorted_probs));
    filtered = filtered / filtered.sum();
    auto sampled_sorted = torch::multinomial(filtered, /*num_samples=*/1).item<int64_t>();
    return sorted_indices.index({sampled_sorted}).item<int64_t>();
  }
  return torch::multinomial(probs, /*num_samples=*/1).item<int64_t>();
}

cverl::rollout::RolloutResponse generate_in_process(
    cverl::torch_backend::CausalLmPolicy& policy,
    const cverl::text::Tokenizer& tokenizer,
    const std::vector<std::string>& prompts,
    const Args& args,
    torch::Device device,
    uint64_t seed) {
  torch::NoGradGuard no_grad;
  torch::manual_seed(static_cast<int64_t>(seed));

  cverl::rollout::RolloutResponse response;
  response.sequences.reserve(prompts.size() * static_cast<size_t>(args.n));

  cverl::text::EncodeOptions prompt_encode;
  prompt_encode.add_bos = false;
  prompt_encode.add_eos = false;
  prompt_encode.max_tokens = static_cast<int32_t>(args.max_prompt_tokens);

  const int32_t pad_id = tokenizer.pad_id() >= 0 ? tokenizer.pad_id() : 0;
  const int32_t eos_id = tokenizer.eos_id();
  const int64_t max_tokens = static_cast<int64_t>(args.max_tokens);

  for (size_t pidx = 0; pidx < prompts.size(); ++pidx) {
    auto prompt_tokens = tokenizer.encode(prompts[pidx], prompt_encode);
    if (prompt_tokens.empty()) {
      prompt_tokens.push_back(pad_id);
    }
    auto prompt_ids = ids_to_tensor(prompt_tokens, device);

    for (uint32_t sample = 0; sample < args.n; ++sample) {
      std::vector<int32_t> generated;
      generated.reserve(static_cast<size_t>(max_tokens));
      for (int64_t t = 0; t < max_tokens; ++t) {
        std::vector<int32_t> response_context = generated;
        response_context.push_back(pad_id);
        auto response_ids = ids_to_tensor(response_context, device);
        auto logits = policy.forward(prompt_ids, response_ids).index({0, -1});
        int64_t next = sample_next_token(logits, args.temperature, args.top_p);
        if (next < 0 || next >= tokenizer.vocab_size()) {
          next = tokenizer.unk_id() >= 0 ? tokenizer.unk_id() : pad_id;
        }
        if (eos_id >= 0 && next == eos_id) {
          break;
        }
        generated.push_back(static_cast<int32_t>(next));
      }

      cverl::rollout::RolloutSequence seq;
      seq.prompt_index = static_cast<uint32_t>(pidx);
      seq.sample_index = sample;
      seq.text = tokenizer.decode(generated, true);
      seq.token_ids = generated;
      seq.finish_reason = generated.size() >= static_cast<size_t>(max_tokens) ? "length" : "stop";
      response.sequences.push_back(std::move(seq));
    }
  }
  response.metrics["rollout_backend"] = "in_process_policy";
  return response;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);
    torch::manual_seed(static_cast<int64_t>(args.seed));

    cverl::data::JsonlDatasetOptions dataset_opts;
    dataset_opts.path = args.dataset;
    auto data = cverl::data::load_prompt_answer_jsonl(dataset_opts);
    if (data.empty()) {
      throw std::runtime_error("dataset is empty");
    }

    std::unique_ptr<cverl::text::Tokenizer> tokenizer;
    if (args.tokenizer_kind == "byte") {
      tokenizer = std::make_unique<cverl::text::ByteTokenizer>();
    } else if (args.tokenizer_kind == "hf") {
      if (args.tokenizer_path.empty()) {
        throw std::invalid_argument("--tokenizer hf requires --tokenizer-path");
      }
      cverl::text::HfBpeTokenizerOptions opts;
      opts.tokenizer_json_path = args.tokenizer_path;
      tokenizer = std::make_unique<cverl::text::HfBpeTokenizer>(opts);
    } else {
      throw std::invalid_argument("--tokenizer must be byte|hf");
    }
    int32_t vocab_size = tokenizer->vocab_size();
    int32_t pad_id = tokenizer->pad_id() >= 0 ? tokenizer->pad_id() : 0;

    cverl::reward::Gsm8kRewardOptions reward_opts;
    reward_opts.method = args.reward_method;

    cverl::torch_backend::CausalLmPolicyOptions policy_opts;
    policy_opts.pad_id = pad_id;
    if (args.policy_kind == "tiny") {
      policy_opts.kind = cverl::torch_backend::CausalLmPolicyOptions::Kind::kTiny;
      policy_opts.tiny_vocab_size = vocab_size;
      policy_opts.tiny_hidden_dim = args.hidden_dim;
    } else {
      policy_opts.kind = cverl::torch_backend::CausalLmPolicyOptions::Kind::kQwen3_5;
      policy_opts.qwen_model_dir = args.model_dir;
      policy_opts.qwen_max_layers = args.qwen_max_layers;
    }
    auto policy = cverl::torch_backend::make_causal_lm_policy(policy_opts);
    torch::Device run_device(torch::kCPU);
    if (args.device == "cuda") {
      run_device = torch::Device(torch::kCUDA, 0);
    } else if (args.device != "cpu") {
      throw std::invalid_argument("--device must be cpu|cuda");
    }
    policy->to_device(run_device);

    torch::optim::AdamW optimizer(
        policy->parameters(),
        torch::optim::AdamWOptions(args.learning_rate).weight_decay(args.weight_decay));

    // Snapshot the initial policy as the frozen reference for KL penalty.
    // For production RLHF runs this is typically the SFT model loaded
    // separately. This trainer defaults to the initial actor as reference.
    auto ref_policy = policy->clone_as_reference();
    ref_policy->to_device(run_device);

    std::cout << "step,rollout,total_seq,mean_reward,success_rate,no_answer,loss,pg_loss,kl_loss,ppo_kl,clipfrac,param_delta,seconds\n";

    for (int64_t step = 1; step <= args.steps; ++step) {
      // Build the per-step batch from the dataset.
      int64_t take = std::min<int64_t>(args.prompts_per_batch, static_cast<int64_t>(data.size()));
      std::vector<std::string> prompts;
      std::vector<std::string> ground_truths;
      prompts.reserve(take);
      ground_truths.reserve(take);
      for (int64_t i = 0; i < take; ++i) {
        size_t idx = static_cast<size_t>(((step - 1) * args.prompts_per_batch + i) % data.size());
        prompts.push_back(data[idx].prompt);
        ground_truths.push_back(data[idx].answer);
      }

      auto t0 = std::chrono::steady_clock::now();
      auto resp = generate_in_process(
          *policy, *tokenizer, prompts, args, run_device, args.seed + static_cast<uint64_t>(step));
      auto t1 = std::chrono::steady_clock::now();

      cverl::rollout::RolloutBatchOptions batch_opts;
      batch_opts.max_prompt_tokens = args.max_prompt_tokens;
      batch_opts.max_response_tokens = args.max_response_tokens;
      auto batch = cverl::rollout::build_gsm8k_rollout_batch(
          resp, prompts, ground_truths, reward_opts, *tokenizer, batch_opts);

      // RolloutBatch tensors are built on CPU; move every trainer-side tensor
      // onto the active device before any forward / loss math so policies
      // running on CUDA don't pay a host->device hop on every op.
      auto prompt_ids = batch.prompt_ids.to(run_device);
      auto response_ids = batch.response_ids.to(run_device);
      auto response_mask = batch.response_mask.to(run_device);
      auto token_rewards = batch.token_rewards.to(run_device);
      auto group_ids = batch.group_ids.to(run_device);

      // GRPO advantages from the rule reward.
      torch::Tensor returns;
      torch::Tensor advantages = cverl::torch_backend::grpo_outcome_advantage(
          token_rewards, response_mask, group_ids, 1.0e-6, true, &returns);

      // Old log probs under the current policy (frozen for this step) and
      // reference log probs (frozen forever) for KL penalty.
      torch::Tensor old_log_probs;
      torch::Tensor ref_log_probs;
      {
        torch::NoGradGuard no_grad;
        torch::Tensor logits = policy->forward(prompt_ids, response_ids);
        old_log_probs = cverl::torch_backend::response_log_probs(logits, response_ids).detach();
        if (args.kl_coef > 0.0) {
          torch::Tensor ref_logits = ref_policy->forward(prompt_ids, response_ids);
          ref_log_probs = cverl::torch_backend::response_log_probs(ref_logits, response_ids).detach();
        }
      }

      std::vector<torch::Tensor> params_before;
      params_before.reserve(policy->parameters().size());
      {
        torch::NoGradGuard no_grad;
        for (const auto& param : policy->parameters()) {
          params_before.push_back(param.detach().clone());
        }
      }

      double last_loss = 0.0;
      double last_pg_loss = 0.0;
      double last_kl_loss = 0.0;
      double last_kl = 0.0;
      double last_clipfrac = 0.0;
      for (int64_t epoch = 0; epoch < args.ppo_epochs; ++epoch) {
        torch::Tensor logits = policy->forward(prompt_ids, response_ids);
        torch::Tensor log_probs = cverl::torch_backend::response_log_probs(logits, response_ids);
        auto loss = cverl::torch_backend::ppo_clipped_loss(
            old_log_probs, log_probs, advantages, response_mask,
            args.clip_ratio, -1.0, -1.0, 3.0, CVERL_LOSS_AGG_TOKEN_MEAN);
        torch::Tensor total_loss = loss.pg_loss;
        torch::Tensor kl_loss_value = torch::zeros({}, total_loss.options());
        if (args.kl_coef > 0.0) {
          torch::Tensor kl_token = cverl::torch_backend::kl_penalty(
              log_probs, ref_log_probs, args.kl_penalty_mode);
          kl_loss_value = cverl::torch_backend::masked_mean(kl_token, response_mask);
          total_loss = total_loss + args.kl_coef * kl_loss_value;
        }
        optimizer.zero_grad();
        total_loss.backward();
        optimizer.step();

        last_loss = total_loss.item<double>();
        last_pg_loss = loss.pg_loss.item<double>();
        last_kl_loss = kl_loss_value.item<double>();
        last_kl = loss.ppo_kl.item<double>();
        last_clipfrac = loss.pg_clipfrac.item<double>();
      }

      double param_delta = 0.0;
      {
        torch::NoGradGuard no_grad;
        auto params_after = policy->parameters();
        for (size_t i = 0; i < params_before.size(); ++i) {
          param_delta += (params_after[i].detach() - params_before[i]).abs().sum().item<double>();
        }
      }

      if (!args.export_dir.empty() && args.export_every > 0 && step % args.export_every == 0) {
        if (!policy->supports_hf_checkpoint_export()) {
          throw std::runtime_error("--export-dir is only supported by HF-style policies such as --policy qwen");
        }
        const std::string path = step_export_path(args.export_dir, step);
        policy->save_hf_checkpoint(path, args.export_dtype);
        std::cerr << "exported_hf_checkpoint=" << path << "\n";
      }

      // Reporting.
      double mean_reward = batch.scalar_rewards.mean().item<double>();
      double success_rate = (batch.scalar_rewards >= reward_opts.correct_score - 1e-6).to(torch::kFloat32).mean().item<double>();
      int64_t no_answer = 0;
      for (const auto& text : batch.response_texts) {
        if (!cverl::reward::extract_gsm8k_answer(text, reward_opts).has_value()) {
          ++no_answer;
        }
      }
      double seconds = std::chrono::duration<double>(t1 - t0).count();

      std::cout << step << ","
                << "in_process_policy" << ","
                << batch.scalar_rewards.size(0) << ","
                << std::fixed << std::setprecision(6) << mean_reward << ","
                << success_rate << ","
                << no_answer << ","
                << last_loss << ","
                << last_pg_loss << ","
                << last_kl_loss << ","
                << last_kl << ","
                << last_clipfrac << ","
                << param_delta << ","
                << seconds << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "gsm8k_grpo_trainer failed: " << e.what() << "\n";
    return 1;
  }
}
