#include <torch/extension.h>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cverl/reward/gsm8k.h"
#include "cverl/rollout/rollout_batch.h"
#include "cverl/rollout/transport.h"
#include "cverl/text/hf_bpe_tokenizer.h"
#include "cverl/torch/causal_lm_policy.h"
#include "cverl/torch/core_algos_torch.h"
#include "cverl/torch/tiny_causal_policy.h"

namespace py = pybind11;

namespace {

std::string restore_hf_name(std::string registered_name) {
  std::replace(registered_name.begin(), registered_name.end(), '/', '.');
  return registered_name;
}

torch::Device parse_device(const std::string& device) {
  if (device == "cpu") {
    return torch::Device(torch::kCPU);
  }
  if (device == "cuda") {
    return torch::Device(torch::kCUDA, 0);
  }
  if (device.rfind("cuda:", 0) == 0) {
    return torch::Device(device);
  }
  throw std::invalid_argument("unsupported device: " + device);
}

std::string dtype_name(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kFloat32:
      return "float32";
    case torch::kFloat16:
      return "float16";
    case torch::kBFloat16:
      return "bfloat16";
    default:
      throw std::invalid_argument("unsupported dtype for vLLM weight sync");
  }
}

torch::ScalarType parse_param_dtype(const std::string& dtype) {
  if (dtype == "float32" || dtype == "fp32" || dtype == "f32") {
    return torch::kFloat32;
  }
  if (dtype == "bfloat16" || dtype == "bf16") {
    return torch::kBFloat16;
  }
  if (dtype == "float16" || dtype == "fp16" || dtype == "f16") {
    return torch::kFloat16;
  }
  throw std::invalid_argument("param_dtype must be float32|bfloat16|float16");
}

class QwenPolicyHandle {
 public:
  QwenPolicyHandle(const std::string& model_dir,
                   const std::string& tokenizer_path,
                   const std::string& device,
                   const std::string& param_dtype,
                   int64_t max_layers) {
    cverl::text::HfBpeTokenizerOptions tok_opts;
    tok_opts.tokenizer_json_path = tokenizer_path.empty()
        ? (std::filesystem::path(model_dir) / "tokenizer.json").string()
        : tokenizer_path;
    tokenizer_ = std::make_unique<cverl::text::HfBpeTokenizer>(tok_opts);

    cverl::torch_backend::CausalLmPolicyOptions opts;
    opts.kind = cverl::torch_backend::CausalLmPolicyOptions::Kind::kQwen3_5;
    opts.qwen_model_dir = model_dir;
    opts.qwen_max_layers = max_layers;
    opts.pad_id = tokenizer_->pad_id() >= 0 ? tokenizer_->pad_id() : 0;
    opts.param_dtype = parse_param_dtype(param_dtype);
    policy_ = cverl::torch_backend::make_causal_lm_policy(opts);
    to_device(device);
  }

  void to_device(const std::string& device) {
    policy_->to_device(parse_device(device));
  }

  std::vector<std::pair<std::string, torch::Tensor>> named_parameters() {
    std::vector<std::pair<std::string, torch::Tensor>> out;
    auto named = policy_->named_parameters(/*recurse=*/true);
    out.reserve(named.size());
    for (const auto& kv : named) {
      out.emplace_back(restore_hf_name(kv.key()), kv.value());
    }
    return out;
  }

  py::dict update_info(bool packed,
                       int64_t packed_buffer_size_bytes,
                       int64_t packed_num_buffers) {
    py::list names;
    py::list dtype_names;
    py::list shapes;
    for (const auto& item : named_parameters()) {
      names.append(item.first);
      dtype_names.append(dtype_name(item.second.scalar_type()));
      py::list shape;
      for (int64_t dim : item.second.sizes()) {
        shape.append(dim);
      }
      shapes.append(shape);
    }
    py::dict out;
    out["names"] = names;
    out["dtype_names"] = dtype_names;
    out["shapes"] = shapes;
    out["packed"] = packed;
    out["packed_buffer_size_bytes"] = packed_buffer_size_bytes;
    out["packed_num_buffers"] = packed_num_buffers;
    out["is_checkpoint_format"] = false;
    return out;
  }

  py::dict gsm8k_grpo_update(const std::vector<std::string>& prompts,
                             const std::vector<std::string>& responses,
                             const std::vector<std::string>& answers,
                             const std::vector<uint32_t>& prompt_indices,
                             const std::vector<uint32_t>& sample_indices,
                             int64_t max_prompt_tokens,
                             int64_t max_response_tokens,
                             int64_t ppo_epochs,
                             double learning_rate,
                             double weight_decay,
                             double clip_ratio,
                             double kl_coef,
                             const std::string& reward_method,
                             bool measure_param_delta) {
    if (responses.size() != prompt_indices.size() || responses.size() != sample_indices.size()) {
      throw std::invalid_argument("responses, prompt_indices, and sample_indices size mismatch");
    }
    if (prompts.size() != answers.size()) {
      throw std::invalid_argument("prompts and answers size mismatch");
    }
    if (responses.empty()) {
      throw std::invalid_argument("empty rollout response");
    }
    if (max_prompt_tokens <= 0 || max_response_tokens <= 0 || ppo_epochs <= 0) {
      throw std::invalid_argument("invalid GRPO update shape");
    }

    int64_t total_seq = static_cast<int64_t>(responses.size());
    double mean_reward = 0.0;
    double success_rate = 0.0;
    double adv_abs_sum = 0.0;
    double last_loss = 0.0;
    double last_pg_loss = 0.0;
    double last_kl_loss = 0.0;
    double last_kl = 0.0;
    double last_clipfrac = 0.0;
    double param_delta = 0.0;

    {
    py::gil_scoped_release no_gil;

    ensure_optimizer(learning_rate, weight_decay);
    if (kl_coef > 0.0) {
      ensure_reference_policy();
    }

    cverl::rollout::RolloutResponse rollout;
    rollout.sequences.reserve(responses.size());
    for (size_t i = 0; i < responses.size(); ++i) {
      cverl::rollout::RolloutSequence seq;
      seq.prompt_index = prompt_indices[i];
      seq.sample_index = sample_indices[i];
      seq.text = responses[i];
      seq.finish_reason = "stop";
      rollout.sequences.push_back(std::move(seq));
    }

    cverl::reward::Gsm8kRewardOptions reward_opts;
    if (reward_method == "strict") {
      reward_opts.method = cverl::reward::Gsm8kExtractionMethod::Strict;
    } else if (reward_method == "flexible") {
      reward_opts.method = cverl::reward::Gsm8kExtractionMethod::Flexible;
    } else {
      throw std::invalid_argument("reward_method must be strict|flexible");
    }

    cverl::rollout::RolloutBatchOptions batch_opts;
    batch_opts.max_prompt_tokens = max_prompt_tokens;
    batch_opts.max_response_tokens = max_response_tokens;
    auto batch = cverl::rollout::build_gsm8k_rollout_batch(
        rollout, prompts, answers, reward_opts, *tokenizer_, batch_opts);

    auto device = current_device();
    auto prompt_ids = batch.prompt_ids.to(device);
    auto response_ids = batch.response_ids.to(device);
    auto response_mask = batch.response_mask.to(device);
    auto token_rewards = batch.token_rewards.to(device);
    auto group_ids = batch.group_ids.to(device);

    torch::Tensor returns;
    auto advantages = cverl::torch_backend::grpo_outcome_advantage(
        token_rewards, response_mask, group_ids, 1.0e-6, true, &returns);
    adv_abs_sum = advantages.abs().sum().item<double>();
    mean_reward = batch.scalar_rewards.mean().item<double>();
    success_rate =
        (batch.scalar_rewards >= reward_opts.correct_score - 1e-6).to(torch::kFloat32).mean().item<double>();

    torch::Tensor old_log_probs;
    torch::Tensor ref_log_probs;
    {
      torch::NoGradGuard no_grad;
      auto logits = policy_->forward(prompt_ids, response_ids);
      old_log_probs = cverl::torch_backend::response_log_probs(logits, response_ids).detach();
      if (kl_coef > 0.0) {
        auto ref_logits = ref_policy_->forward(prompt_ids, response_ids);
        ref_log_probs = cverl::torch_backend::response_log_probs(ref_logits, response_ids).detach();
      }
    }

    std::vector<torch::Tensor> params_before;
    if (measure_param_delta) {
      torch::NoGradGuard no_grad;
      for (const auto& param : policy_->parameters()) {
        params_before.push_back(param.detach().clone());
      }
    }

    for (int64_t epoch = 0; epoch < ppo_epochs; ++epoch) {
      auto logits = policy_->forward(prompt_ids, response_ids);
      auto log_probs = cverl::torch_backend::response_log_probs(logits, response_ids);
      auto loss = cverl::torch_backend::ppo_clipped_loss(
          old_log_probs, log_probs, advantages, response_mask,
          clip_ratio, -1.0, -1.0, 3.0, CVERL_LOSS_AGG_TOKEN_MEAN);
      torch::Tensor total_loss = loss.pg_loss;
      torch::Tensor kl_loss_value = torch::zeros({}, total_loss.options());
      if (kl_coef > 0.0) {
        auto kl_token = cverl::torch_backend::kl_penalty(log_probs, ref_log_probs, CVERL_KL_K1);
        kl_loss_value = cverl::torch_backend::masked_mean(kl_token, response_mask);
        total_loss = total_loss + kl_coef * kl_loss_value;
      }
      optimizer_->zero_grad();
      total_loss.backward();
      optimizer_->step();

      last_loss = total_loss.template item<double>();
      last_pg_loss = loss.pg_loss.template item<double>();
      last_kl_loss = kl_loss_value.template item<double>();
      last_kl = loss.ppo_kl.template item<double>();
      last_clipfrac = loss.pg_clipfrac.template item<double>();
    }

    if (measure_param_delta) {
      torch::NoGradGuard no_grad;
      auto params_after = policy_->parameters();
      for (size_t i = 0; i < params_before.size(); ++i) {
        param_delta += (params_after[i].detach() - params_before[i]).abs().sum().item<double>();
      }
    }
    }

    py::dict out;
    out["total_seq"] = total_seq;
    out["mean_reward"] = mean_reward;
    out["success_rate"] = success_rate;
    out["adv_abs_sum"] = adv_abs_sum;
    out["loss"] = last_loss;
    out["pg_loss"] = last_pg_loss;
    out["kl_loss"] = last_kl_loss;
    out["ppo_kl"] = last_kl;
    out["clipfrac"] = last_clipfrac;
    out["param_delta"] = param_delta;
    return out;
  }

 private:
  torch::Device current_device() {
    auto named = policy_->named_parameters(/*recurse=*/true);
    if (named.is_empty()) {
      return torch::Device(torch::kCPU);
    }
    return named.front().value().device();
  }

  void ensure_optimizer(double learning_rate, double weight_decay) {
    if (optimizer_ != nullptr && learning_rate == optimizer_lr_ && weight_decay == optimizer_weight_decay_) {
      return;
    }
    optimizer_ = std::make_unique<torch::optim::AdamW>(
        policy_->parameters(),
        torch::optim::AdamWOptions(learning_rate).weight_decay(weight_decay));
    optimizer_lr_ = learning_rate;
    optimizer_weight_decay_ = weight_decay;
  }

  void ensure_reference_policy() {
    if (ref_policy_ != nullptr) {
      return;
    }
    ref_policy_ = policy_->clone_as_reference();
    ref_policy_->to_device(current_device());
  }

  std::unique_ptr<cverl::text::HfBpeTokenizer> tokenizer_;
  std::shared_ptr<cverl::torch_backend::CausalLmPolicy> policy_;
  std::shared_ptr<cverl::torch_backend::CausalLmPolicy> ref_policy_;
  std::unique_ptr<torch::optim::AdamW> optimizer_;
  double optimizer_lr_ = 0.0;
  double optimizer_weight_decay_ = 0.0;
};

}  // namespace

PYBIND11_MODULE(_cverl_vllm_bridge, m) {
  py::class_<QwenPolicyHandle>(m, "QwenPolicy")
      .def(py::init<const std::string&, const std::string&, const std::string&, const std::string&, int64_t>(),
           py::arg("model_dir"),
           py::arg("tokenizer_path") = "",
           py::arg("device") = "cuda:0",
           py::arg("param_dtype") = "float32",
           py::arg("max_layers") = -1)
      .def("to_device", &QwenPolicyHandle::to_device)
      .def("named_parameters", &QwenPolicyHandle::named_parameters)
      .def("update_info", &QwenPolicyHandle::update_info,
           py::arg("packed") = true,
           py::arg("packed_buffer_size_bytes") = 1024LL * 1024LL * 1024LL,
           py::arg("packed_num_buffers") = 2)
      .def("gsm8k_grpo_update", &QwenPolicyHandle::gsm8k_grpo_update,
           py::arg("prompts"),
           py::arg("responses"),
           py::arg("answers"),
           py::arg("prompt_indices"),
           py::arg("sample_indices"),
           py::arg("max_prompt_tokens") = 256,
           py::arg("max_response_tokens") = 256,
           py::arg("ppo_epochs") = 1,
           py::arg("learning_rate") = 3.0e-6,
           py::arg("weight_decay") = 0.0,
           py::arg("clip_ratio") = 0.2,
           py::arg("kl_coef") = 0.0,
           py::arg("reward_method") = "flexible",
           py::arg("measure_param_delta") = false);
}
