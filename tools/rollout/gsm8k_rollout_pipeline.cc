// gsm8k_rollout_pipeline
//
// End-to-end skeleton for the rollout half of GRPO/PPO:
//   GSM8K JSONL  ->  RolloutTransport  ->  rule reward  ->  per-prompt stats
//
// Transport selection:
//   --transport http       call vLLM/SGLang at --base-url
//   --transport loopback   echo the prompt back (CPU smoke, no server)
//
// The output is intentionally small and machine-parseable so the next step
// (logprob + GRPO update) can be slotted in without touching this binary.
#include "cverl/data/hf_dataset.h"
#include "cverl/reward/gsm8k.h"
#include "cverl/rollout/http_transport.h"
#include "cverl/rollout/transport.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string dataset;
  std::string transport = "loopback";
  std::string base_url;
  std::string endpoint = "completions";
  std::string api_key;
  std::string model;
  std::string system_prompt;
  std::string loopback_suffix = " #### 0";
  int64_t prompts = 4;
  uint32_t n = 4;
  uint32_t max_tokens = 256;
  double temperature = 1.0;
  double top_p = 1.0;
  uint64_t seed = 1;
  cverl::reward::Gsm8kExtractionMethod reward_method = cverl::reward::Gsm8kExtractionMethod::Strict;
  bool verbose = false;
  long connect_timeout = 10;
  long total_timeout = 600;
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
    } else if (a == "--transport") {
      args.transport = require_value(i, argc, argv, "--transport");
    } else if (a == "--base-url") {
      args.base_url = require_value(i, argc, argv, "--base-url");
    } else if (a == "--endpoint") {
      args.endpoint = require_value(i, argc, argv, "--endpoint");
    } else if (a == "--api-key") {
      args.api_key = require_value(i, argc, argv, "--api-key");
    } else if (a == "--model") {
      args.model = require_value(i, argc, argv, "--model");
    } else if (a == "--system-prompt") {
      args.system_prompt = require_value(i, argc, argv, "--system-prompt");
    } else if (a == "--loopback-suffix") {
      args.loopback_suffix = require_value(i, argc, argv, "--loopback-suffix");
    } else if (a == "--prompts") {
      args.prompts = std::strtoll(require_value(i, argc, argv, "--prompts"), nullptr, 10);
    } else if (a == "--n") {
      args.n = static_cast<uint32_t>(std::strtoul(require_value(i, argc, argv, "--n"), nullptr, 10));
    } else if (a == "--max-tokens") {
      args.max_tokens = static_cast<uint32_t>(std::strtoul(require_value(i, argc, argv, "--max-tokens"), nullptr, 10));
    } else if (a == "--temperature") {
      args.temperature = std::strtod(require_value(i, argc, argv, "--temperature"), nullptr);
    } else if (a == "--top-p") {
      args.top_p = std::strtod(require_value(i, argc, argv, "--top-p"), nullptr);
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
    } else if (a == "--connect-timeout") {
      args.connect_timeout = std::strtol(require_value(i, argc, argv, "--connect-timeout"), nullptr, 10);
    } else if (a == "--total-timeout") {
      args.total_timeout = std::strtol(require_value(i, argc, argv, "--total-timeout"), nullptr, 10);
    } else if (a == "--verbose") {
      args.verbose = true;
    } else {
      throw std::invalid_argument("unknown argument: " + a);
    }
  }
  if (args.dataset.empty()) {
    throw std::invalid_argument("--dataset is required");
  }
  if (args.prompts <= 0 || args.n == 0) {
    throw std::invalid_argument("--prompts must be > 0 and --n must be >= 1");
  }
  if (args.transport == "http" && args.base_url.empty()) {
    throw std::invalid_argument("--transport http requires --base-url");
  }
  return args;
}

std::unique_ptr<cverl::rollout::RolloutTransport> build_transport(const Args& args) {
  if (args.transport == "loopback") {
    cverl::rollout::LoopbackRolloutTransport::Options opts;
    opts.completion_suffix = args.loopback_suffix;
    return std::make_unique<cverl::rollout::LoopbackRolloutTransport>(opts);
  }
  if (args.transport == "http") {
    cverl::rollout::HttpRolloutOptions opts;
    opts.base_url = args.base_url;
    opts.endpoint = args.endpoint;
    opts.api_key = args.api_key;
    opts.model = args.model;
    opts.connect_timeout_seconds = args.connect_timeout;
    opts.total_timeout_seconds = args.total_timeout;
    opts.verbose = args.verbose;
    opts.system_prompt = args.system_prompt;
    return std::make_unique<cverl::rollout::HttpRolloutTransport>(std::move(opts));
  }
  throw std::invalid_argument("unsupported transport: " + args.transport);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);

    cverl::data::JsonlDatasetOptions dataset_options;
    dataset_options.path = args.dataset;
    dataset_options.max_examples = args.prompts;
    auto data = cverl::data::load_prompt_answer_jsonl(dataset_options);
    if (data.empty()) {
      throw std::runtime_error("dataset is empty");
    }
    int64_t take = std::min<int64_t>(args.prompts, static_cast<int64_t>(data.size()));

    cverl::rollout::RolloutRequest request;
    request.prompts.reserve(take);
    std::vector<std::string> ground_truths;
    ground_truths.reserve(take);
    for (int64_t i = 0; i < take; ++i) {
      request.prompts.push_back(data[static_cast<size_t>(i)].prompt);
      ground_truths.push_back(data[static_cast<size_t>(i)].answer);
    }
    request.n = args.n;
    request.max_tokens = args.max_tokens;
    request.temperature = args.temperature;
    request.top_p = args.top_p;
    request.seed = args.seed;
    request.model = args.model;
    request.return_logprobs = false;

    auto transport = build_transport(args);
    auto t0 = std::chrono::steady_clock::now();
    auto response = transport->generate(request);
    auto t1 = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(t1 - t0).count();

    cverl::reward::Gsm8kRewardOptions reward_opts;
    reward_opts.method = args.reward_method;

    std::vector<double> rewards;
    rewards.reserve(response.sequences.size());
    int64_t correct = 0;
    int64_t no_answer = 0;
    for (const auto& seq : response.sequences) {
      if (seq.prompt_index >= ground_truths.size()) {
        throw std::runtime_error("rollout prompt_index out of range");
      }
      double r = cverl::reward::compute_gsm8k_reward(seq.text, ground_truths[seq.prompt_index], reward_opts);
      rewards.push_back(r);
      if (r >= reward_opts.correct_score) {
        ++correct;
      }
      if (!cverl::reward::extract_gsm8k_answer(seq.text, reward_opts).has_value()) {
        ++no_answer;
      }
    }
    double mean_reward = rewards.empty() ? 0.0 : std::accumulate(rewards.begin(), rewards.end(), 0.0) / rewards.size();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "transport," << transport->name() << "\n";
    std::cout << "prompts," << take << "\n";
    std::cout << "samples_per_prompt," << args.n << "\n";
    std::cout << "total_sequences," << response.sequences.size() << "\n";
    std::cout << "correct," << correct << "\n";
    std::cout << "no_answer," << no_answer << "\n";
    std::cout << "mean_reward," << mean_reward << "\n";
    std::cout << "rollout_seconds," << seconds << "\n";
    std::cout << "sequences_per_second,"
              << (seconds > 0 ? static_cast<double>(response.sequences.size()) / seconds : 0.0) << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "gsm8k_rollout_pipeline failed: " << e.what() << "\n";
    return 1;
  }
}
