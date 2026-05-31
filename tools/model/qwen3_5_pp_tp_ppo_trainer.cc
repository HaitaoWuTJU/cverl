#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#include <torch/torch.h>

#include "cverl/data/hf_dataset.h"
#include "cverl/distributed/nccl_collectives.h"
#include "cverl/distributed/optimizer_sharding.h"
#include "cverl/distributed/parallel_ops.h"
#include "cverl/distributed/pipeline.h"
#include "cverl/distributed/topology.h"
#include "cverl/distributed/weight_sync.h"
#include "cverl/model/hf_model_loader.h"
#include "cverl/model/qwen3_5_text.h"
#include "cverl/reward/gsm8k.h"
#include "cverl/rollout/gpu_ipc_rollout_batch.h"
#include "cverl/rollout/nccl_gpu_batch_transport.h"
#include "cverl/rollout/rollout_batch.h"
#include "cverl/rollout/transport.h"
#include "cverl/text/hf_bpe_tokenizer.h"
#include "cverl/text/tokenizer.h"
#include "cverl/torch/core_algos_torch.h"
#include "cverl/torch/fp32_master_adamw.h"
#include "cverl/torch/tiny_causal_policy.h"

namespace {

int64_t arg_i64(int argc, char** argv, const std::string& flag, int64_t fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == flag) {
      return std::stoll(argv[i + 1]);
    }
  }
  return fallback;
}

double arg_f64(int argc, char** argv, const std::string& flag, double fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == flag) {
      return std::stod(argv[i + 1]);
    }
  }
  return fallback;
}

bool arg_bool(int argc, char** argv, const std::string& flag, bool fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == flag) {
      const std::string value = argv[i + 1];
      if (value == "1" || value == "true" || value == "yes" || value == "on") {
        return true;
      }
      if (value == "0" || value == "false" || value == "no" || value == "off") {
        return false;
      }
      throw std::invalid_argument("invalid boolean value for " + flag);
    }
  }
  return fallback;
}

std::string arg_str(int argc, char** argv, const std::string& flag, const std::string& fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == flag) {
      return argv[i + 1];
    }
  }
  return fallback;
}

std::vector<int64_t> full_group(int64_t size) {
  std::vector<int64_t> ranks;
  ranks.reserve(static_cast<size_t>(size));
  for (int64_t i = 0; i < size; ++i) {
    ranks.push_back(i);
  }
  return ranks;
}

std::vector<torch::Tensor> clone_params(const std::vector<torch::Tensor>& params) {
  std::vector<torch::Tensor> out;
  out.reserve(params.size());
  for (const auto& p : params) {
    out.push_back(p.detach().clone());
  }
  return out;
}

std::vector<int64_t> full_index_list(size_t size) {
  std::vector<int64_t> out;
  out.reserve(size);
  for (size_t i = 0; i < size; ++i) {
    out.push_back(static_cast<int64_t>(i));
  }
  return out;
}

std::vector<int64_t> param_index_to_optimizer_position(size_t param_count,
                                                       const std::vector<int64_t>& optimizer_param_indices) {
  std::vector<int64_t> out(param_count, -1);
  for (size_t i = 0; i < optimizer_param_indices.size(); ++i) {
    const int64_t param_index = optimizer_param_indices[i];
    if (param_index < 0 || param_index >= static_cast<int64_t>(param_count)) {
      throw std::runtime_error("optimizer parameter index out of range");
    }
    out[static_cast<size_t>(param_index)] = static_cast<int64_t>(i);
  }
  return out;
}

std::vector<torch::Tensor> select_params(const std::vector<torch::Tensor>& params,
                                         const std::vector<int64_t>& indices) {
  std::vector<torch::Tensor> out;
  out.reserve(indices.size());
  for (int64_t index : indices) {
    if (index < 0 || index >= static_cast<int64_t>(params.size())) {
      throw std::runtime_error("parameter selection index out of range");
    }
    out.push_back(params[static_cast<size_t>(index)]);
  }
  return out;
}

void zero_model_gradients(const std::vector<torch::Tensor>& params) {
  torch::NoGradGuard no_grad;
  for (const auto& p : params) {
    if (p.defined() && p.grad().defined()) {
      p.mutable_grad().zero_();
    }
  }
}

double grad_l2_norm_sq(const std::vector<torch::Tensor>& params) {
  double out = 0.0;
  for (const auto& p : params) {
    if (!p.defined() || !p.grad().defined()) {
      continue;
    }
    out += p.grad().detach().to(torch::kFloat32).pow(2).sum().item<double>();
  }
  return out;
}

double grad_norm_sum(const std::vector<torch::Tensor>& params) {
  double out = 0.0;
  for (const auto& p : params) {
    if (!p.defined() || !p.grad().defined()) {
      continue;
    }
    out += p.grad().detach().to(torch::kFloat32).norm().item<double>();
  }
  return out;
}

void scale_model_gradients(const std::vector<torch::Tensor>& params, double scale) {
  torch::NoGradGuard no_grad;
  for (const auto& p : params) {
    if (p.defined() && p.grad().defined()) {
      p.mutable_grad().mul_(scale);
    }
  }
}

void broadcast_dp_sharded_optimizer_parameters(const std::vector<std::string>& param_names,
                                               const std::vector<torch::Tensor>& params,
                                               const std::vector<int64_t>& owner_by_param,
                                               cverl::distributed::Collectives& dp_comm,
                                               const std::vector<int64_t>& dp_ranks,
                                               int64_t bucket_bytes) {
  if (param_names.size() != params.size() || owner_by_param.size() != params.size()) {
    throw std::runtime_error("DP sharded optimizer broadcast metadata size mismatch");
  }
  for (int64_t owner : dp_ranks) {
    std::vector<cverl::distributed::ParameterView> owned;
    for (size_t i = 0; i < params.size(); ++i) {
      if (owner_by_param[i] == owner) {
        owned.push_back(cverl::distributed::ParameterView{param_names[i], params[i]});
      }
    }
    cverl::distributed::broadcast_parameters_from_root(owned, dp_comm, owner, dp_ranks, bucket_bytes);
  }
}

bool layer_index_from_weight_name(const std::string& name, int64_t* layer_index) {
  const std::string marker = "model.language_model.layers.";
  const size_t pos = name.find(marker);
  if (pos == std::string::npos) {
    return false;
  }
  const size_t begin = pos + marker.size();
  size_t end = begin;
  while (end < name.size() && name[end] >= '0' && name[end] <= '9') {
    ++end;
  }
  if (end == begin || end >= name.size() || name[end] != '.') {
    throw std::runtime_error("invalid Qwen layer weight name: " + name);
  }
  *layer_index = std::stoll(name.substr(begin, end - begin));
  return true;
}

std::vector<std::string> stage_required_weight_names(const cverl::Qwen35TextModel& model,
                                                     int64_t max_layers,
                                                     int64_t layer_begin,
                                                     int64_t layer_end,
                                                     bool is_first_stage,
                                                     bool is_last_stage) {
  std::vector<std::string> out;
  for (const auto& name : model.required_weight_names(max_layers)) {
    int64_t layer_index = -1;
    if (layer_index_from_weight_name(name, &layer_index)) {
      if (layer_index >= layer_begin && layer_index < layer_end) {
        out.push_back(name);
      }
      continue;
    }
    if (name == "model.language_model.embed_tokens.weight") {
      if (is_first_stage || is_last_stage) {
        out.push_back(name);
      }
      continue;
    }
    if (name == "model.language_model.norm.weight") {
      if (is_last_stage) {
        out.push_back(name);
      }
      continue;
    }
    out.push_back(name);
  }
  return out;
}

torch::Tensor shard_dim_if_needed(const torch::Tensor& tensor,
                                  int64_t dim,
                                  int64_t rank,
                                  int64_t world_size) {
  if (world_size == 1) {
    return tensor.contiguous();
  }
  if (dim < 0) {
    dim += tensor.dim();
  }
  if (dim < 0 || dim >= tensor.dim() || tensor.size(dim) % world_size != 0) {
    throw std::runtime_error("cannot shard tensor for TP");
  }
  const int64_t shard = tensor.size(dim) / world_size;
  return tensor.narrow(dim, rank * shard, shard).contiguous();
}

torch::Tensor qwen_megatron_tp_shard(const std::string& name,
                                     const torch::Tensor& full,
                                     const cverl::Qwen35TextConfig& cfg,
                                     int64_t tp_rank,
                                     int64_t tp_size) {
  if (tp_size == 1) {
    return full.contiguous();
  }
  if (name == "model.language_model.embed_tokens.weight") {
    return shard_dim_if_needed(full, 0, tp_rank, tp_size);
  }
  int64_t layer = -1;
  if (!layer_index_from_weight_name(name, &layer)) {
    return full.contiguous();
  }
  if (name.find(".mlp.gate_proj.weight") != std::string::npos ||
      name.find(".mlp.up_proj.weight") != std::string::npos) {
    return shard_dim_if_needed(full, 0, tp_rank, tp_size);
  }
  if (name.find(".mlp.down_proj.weight") != std::string::npos) {
    return shard_dim_if_needed(full, 1, tp_rank, tp_size);
  }

  const std::string layer_type = cfg.layer_types.at(static_cast<size_t>(layer));
  if (layer_type == "full_attention") {
    if (name.find(".self_attn.q_proj.weight") != std::string::npos ||
        name.find(".self_attn.k_proj.weight") != std::string::npos ||
        name.find(".self_attn.v_proj.weight") != std::string::npos) {
      return shard_dim_if_needed(full, 0, tp_rank, tp_size);
    }
    if (name.find(".self_attn.o_proj.weight") != std::string::npos) {
      return shard_dim_if_needed(full, 1, tp_rank, tp_size);
    }
    return full.contiguous();
  }

  if (name.find(".linear_attn.in_proj_qkv.weight") != std::string::npos ||
      name.find(".linear_attn.conv1d.weight") != std::string::npos) {
    const int64_t key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    const int64_t value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    if (key_dim % tp_size != 0 || value_dim % tp_size != 0) {
      throw std::runtime_error("linear attention dimensions are not divisible by TP size");
    }
    const int64_t key_local = key_dim / tp_size;
    const int64_t value_local = value_dim / tp_size;
    const int64_t key_begin = tp_rank * key_local;
    const int64_t value_begin = tp_rank * value_local;
    auto q = full.narrow(0, key_begin, key_local);
    auto k = full.narrow(0, key_dim + key_begin, key_local);
    auto v = full.narrow(0, key_dim * 2 + value_begin, value_local);
    return torch::cat(std::vector<torch::Tensor>{q, k, v}, 0).contiguous();
  }
  if (name.find(".linear_attn.in_proj_z.weight") != std::string::npos) {
    return shard_dim_if_needed(full, 0, tp_rank, tp_size);
  }
  if (name.find(".linear_attn.in_proj_b.weight") != std::string::npos ||
      name.find(".linear_attn.in_proj_a.weight") != std::string::npos ||
      name.find(".linear_attn.A_log") != std::string::npos ||
      name.find(".linear_attn.dt_bias") != std::string::npos) {
    return shard_dim_if_needed(full, 0, tp_rank, tp_size);
  }
  if (name.find(".linear_attn.out_proj.weight") != std::string::npos) {
    return shard_dim_if_needed(full, 1, tp_rank, tp_size);
  }
  return full.contiguous();
}

bool qwen_tp_replicated_parameter(const std::string& name) {
  if (name == "model.language_model.norm.weight") {
    return true;
  }
  if (name.find("input_layernorm.weight") != std::string::npos ||
      name.find("post_attention_layernorm.weight") != std::string::npos ||
      name.find(".self_attn.q_norm.weight") != std::string::npos ||
      name.find(".self_attn.k_norm.weight") != std::string::npos ||
      name.find(".linear_attn.norm.weight") != std::string::npos) {
    return true;
  }
  return false;
}

void sync_tp_replicated_gradients(const std::vector<std::string>& names,
                                  const std::vector<torch::Tensor>& params,
                                  cverl::distributed::ParallelGroup& tp_group,
                                  int64_t bucket_bytes) {
  if (tp_group.world_size == 1) {
    return;
  }
  if (tp_group.collectives == nullptr) {
    throw std::runtime_error("TP replicated gradient sync requires collectives");
  }
  if (names.size() != params.size()) {
    throw std::runtime_error("TP replicated gradient sync name/param size mismatch");
  }
  std::vector<torch::Tensor> replicated_params;
  for (size_t i = 0; i < params.size(); ++i) {
    if (!qwen_tp_replicated_parameter(names[i])) {
      continue;
    }
    auto& p = params[i];
    if (!p.defined() || !p.grad().defined()) {
      continue;
    }
    replicated_params.push_back(p);
  }
  cverl::distributed::data_parallel_sync_gradients(
      replicated_params, *tp_group.collectives, tp_group.ranks, true, bucket_bytes);
}

struct PpPpoBatch {
  std::vector<torch::Tensor> token_ids;
  std::vector<torch::Tensor> advantages;
  std::vector<torch::Tensor> response_masks;
  std::vector<torch::Tensor> old_log_probs;
  std::vector<torch::Tensor> ref_log_probs;
  int64_t rows = 0;
  double mean_reward = 0.0;
  double success_rate = 0.0;
  double adv_abs_sum = 0.0;
};

torch::ScalarType parse_dtype(const std::string& dtype) {
  if (dtype == "float32" || dtype == "fp32") {
    return torch::kFloat32;
  }
  if (dtype == "bfloat16" || dtype == "bf16") {
    return torch::kBFloat16;
  }
  if (dtype == "float16" || dtype == "fp16") {
    return torch::kFloat16;
  }
  throw std::invalid_argument("unsupported dtype");
}

cverl_kl_penalty_t parse_kl_penalty_mode(const std::string& mode) {
  if (mode == "k1") {
    return CVERL_KL_K1;
  }
  if (mode == "abs") {
    return CVERL_KL_ABS;
  }
  if (mode == "k2") {
    return CVERL_KL_K2;
  }
  if (mode == "k3") {
    return CVERL_KL_K3;
  }
  throw std::invalid_argument("--kl-penalty must be k1|abs|k2|k3");
}

std::vector<std::string> json_string_vector(const nlohmann::json& obj, const char* key) {
  if (!obj.contains(key) || !obj.at(key).is_array()) {
    throw std::runtime_error(std::string("rollout JSON missing array: ") + key);
  }
  std::vector<std::string> out;
  out.reserve(obj.at(key).size());
  for (const auto& item : obj.at(key)) {
    out.push_back(item.get<std::string>());
  }
  return out;
}

std::vector<uint32_t> json_u32_vector(const nlohmann::json& obj, const char* key) {
  if (!obj.contains(key) || !obj.at(key).is_array()) {
    throw std::runtime_error(std::string("rollout JSON missing array: ") + key);
  }
  std::vector<uint32_t> out;
  out.reserve(obj.at(key).size());
  for (const auto& item : obj.at(key)) {
    out.push_back(item.get<uint32_t>());
  }
  return out;
}

std::string rollout_json_path_for_step(const std::string& rollout_json,
                                       const std::string& rollout_dir,
                                       int64_t step) {
  if (!rollout_json.empty()) {
    return rollout_json;
  }
  if (rollout_dir.empty()) {
    return "";
  }
  char name[64];
  std::snprintf(name, sizeof(name), "rollout_step_%06lld.json", static_cast<long long>(step + 1));
  return (std::filesystem::path(rollout_dir) / name).string();
}

std::string rollout_ipc_path_for_step(const std::string& rollout_ipc_json,
                                      const std::string& rollout_ipc_dir,
                                      int64_t step) {
  if (!rollout_ipc_json.empty()) {
    return rollout_ipc_json;
  }
  if (rollout_ipc_dir.empty()) {
    return "";
  }
  char name[64];
  std::snprintf(name, sizeof(name), "rollout_ipc_step_%06lld.json", static_cast<long long>(step + 1));
  return (std::filesystem::path(rollout_ipc_dir) / name).string();
}

std::string checkpoint_step_name(int64_t step) {
  std::ostringstream step_name;
  step_name << "step_" << std::setw(6) << std::setfill('0') << (step + 1);
  return step_name.str();
}

void write_text_atomic(const std::filesystem::path& path, const std::string& text) {
  const auto tmp = path.string() + ".tmp";
  {
    std::ofstream out(tmp, std::ios::trunc);
    if (!out) {
      throw std::runtime_error("failed to open temp file for write: " + tmp);
    }
    out << text;
    if (!out) {
      throw std::runtime_error("failed to write temp file: " + tmp);
    }
  }
  std::filesystem::rename(tmp, path);
}

void write_json_atomic(const std::filesystem::path& path, const nlohmann::json& doc) {
  write_text_atomic(path, doc.dump(2) + "\n");
}

void save_rank_checkpoint(const std::string& checkpoint_dir,
                          int64_t step,
                          int64_t global_rank,
                          int64_t world_size,
                          int64_t dp_rank,
                          int64_t pp_rank,
                          int64_t pp_size,
                          int64_t tp_rank,
                          int64_t tp_size,
                          int64_t layer_begin,
                          int64_t layer_end,
                          const std::vector<std::string>& param_names,
                          const std::vector<torch::Tensor>& params,
                          const cverl::torch_backend::Fp32MasterAdamW& optimizer,
                          const std::vector<int64_t>& optimizer_param_indices) {
  if (checkpoint_dir.empty()) {
    return;
  }
  if (param_names.size() != params.size()) {
    throw std::runtime_error("checkpoint parameter name/param size mismatch");
  }
  const auto step_dir = std::filesystem::path(checkpoint_dir) / checkpoint_step_name(step);
  std::filesystem::create_directories(step_dir);

  const bool use_master_weights = optimizer.uses_master_weights();
  const auto& master_params = optimizer.master_parameters();
  const auto& exp_avg = optimizer.exp_avg();
  const auto& exp_avg_sq = optimizer.exp_avg_sq();
  if (exp_avg.size() != optimizer_param_indices.size() ||
      exp_avg_sq.size() != optimizer_param_indices.size() ||
      (use_master_weights && master_params.size() != optimizer_param_indices.size())) {
    throw std::runtime_error("checkpoint optimizer state/index size mismatch");
  }
  const auto param_to_optim_pos = param_index_to_optimizer_position(params.size(), optimizer_param_indices);
  torch::serialize::OutputArchive archive;
  for (size_t i = 0; i < params.size(); ++i) {
    const int64_t optim_pos = param_to_optim_pos[i];
    const torch::Tensor& source =
        (use_master_weights && optim_pos >= 0) ? master_params[static_cast<size_t>(optim_pos)] : params[i];
    archive.write("param_" + std::to_string(i), source.detach().to(torch::kCPU));
  }
  for (size_t i = 0; i < optimizer_param_indices.size(); ++i) {
    archive.write("optim_exp_avg_" + std::to_string(i), exp_avg[i].detach().to(torch::kCPU));
    archive.write("optim_exp_avg_sq_" + std::to_string(i), exp_avg_sq[i].detach().to(torch::kCPU));
  }
  archive.write("optim_step", torch::tensor(optimizer.step_count(), torch::kInt64));

  const std::string rank_file = "rank_" + std::to_string(global_rank) + ".pt";
  const std::string rank_meta_file = "rank_" + std::to_string(global_rank) + ".json";
  const auto rank_path = step_dir / rank_file;
  const auto rank_tmp_path = step_dir / (rank_file + ".tmp");
  archive.save_to(rank_tmp_path.string());
  std::filesystem::rename(rank_tmp_path, rank_path);

  nlohmann::json rank_meta;
  rank_meta["global_rank"] = global_rank;
  rank_meta["data_rank"] = dp_rank;
  rank_meta["pipeline_rank"] = pp_rank;
  rank_meta["tensor_rank"] = tp_rank;
  rank_meta["layer_begin"] = layer_begin;
  rank_meta["layer_end"] = layer_end;
  rank_meta["param_file"] = rank_file;
  rank_meta["optimizer_step"] = optimizer.step_count();
  rank_meta["use_master_weights"] = use_master_weights;
  rank_meta["optimizer_param_indices"] = optimizer_param_indices;
  rank_meta["parameters"] = nlohmann::json::array();
  for (size_t i = 0; i < params.size(); ++i) {
    nlohmann::json item;
    item["index"] = i;
    item["name"] = param_names[i];
    item["dtype"] = c10::toString(params[i].scalar_type());
    item["shape"] = std::vector<int64_t>(params[i].sizes().begin(), params[i].sizes().end());
    item["checkpoint_key"] = "param_" + std::to_string(i);
    const int64_t optim_pos = param_to_optim_pos[i];
    item["optimizer_position"] = optim_pos;
    if (optim_pos >= 0) {
      item["optimizer_exp_avg_key"] = "optim_exp_avg_" + std::to_string(optim_pos);
      item["optimizer_exp_avg_sq_key"] = "optim_exp_avg_sq_" + std::to_string(optim_pos);
    }
    rank_meta["parameters"].push_back(std::move(item));
  }
  write_json_atomic(step_dir / rank_meta_file, rank_meta);
}

void save_rank_flat_checkpoint(const std::string& checkpoint_dir,
                               int64_t step,
                               int64_t global_rank,
                               int64_t world_size,
                               int64_t dp_rank,
                               int64_t pp_rank,
                               int64_t pp_size,
                               int64_t tp_rank,
                               int64_t tp_size,
                               int64_t layer_begin,
                               int64_t layer_end,
                               const std::vector<std::string>& param_names,
                               const std::vector<torch::Tensor>& params,
                               const cverl::distributed::FlatParameterShard& flat_param_shard,
                               const cverl::torch_backend::FlatAdamW& optimizer) {
  if (checkpoint_dir.empty()) {
    return;
  }
  if (param_names.size() != params.size()) {
    throw std::runtime_error("flat checkpoint parameter name/param size mismatch");
  }
  const auto step_dir = std::filesystem::path(checkpoint_dir) / checkpoint_step_name(step);
  std::filesystem::create_directories(step_dir);
  if (!flat_param_shard.shard.defined() ||
      optimizer.parameter_shard().numel() != flat_param_shard.shard.numel() ||
      optimizer.exp_avg().numel() != flat_param_shard.shard.numel() ||
      optimizer.exp_avg_sq().numel() != flat_param_shard.shard.numel()) {
    throw std::runtime_error("flat checkpoint optimizer shard size mismatch");
  }

  torch::serialize::OutputArchive archive;
  for (size_t i = 0; i < params.size(); ++i) {
    archive.write("param_" + std::to_string(i), params[i].detach().to(torch::kCPU));
  }
  archive.write("flat_param_shard", optimizer.parameter_shard().detach().to(torch::kCPU));
  archive.write("flat_exp_avg", optimizer.exp_avg().detach().to(torch::kCPU));
  archive.write("flat_exp_avg_sq", optimizer.exp_avg_sq().detach().to(torch::kCPU));
  archive.write("optim_step", torch::tensor(optimizer.step_count(), torch::kInt64));

  const std::string rank_file = "rank_" + std::to_string(global_rank) + ".pt";
  const std::string rank_meta_file = "rank_" + std::to_string(global_rank) + ".json";
  const auto rank_path = step_dir / rank_file;
  const auto rank_tmp_path = step_dir / (rank_file + ".tmp");
  archive.save_to(rank_tmp_path.string());
  std::filesystem::rename(rank_tmp_path, rank_path);

  nlohmann::json rank_meta;
  rank_meta["global_rank"] = global_rank;
  rank_meta["data_rank"] = dp_rank;
  rank_meta["pipeline_rank"] = pp_rank;
  rank_meta["tensor_rank"] = tp_rank;
  rank_meta["layer_begin"] = layer_begin;
  rank_meta["layer_end"] = layer_end;
  rank_meta["param_file"] = rank_file;
  rank_meta["optimizer_step"] = optimizer.step_count();
  rank_meta["optimizer_kind"] = "FlatAdamW";
  rank_meta["optimizer_param_indices"] = full_index_list(params.size());
  rank_meta["flat_parameter_shard"] = {
      {"original_numel", flat_param_shard.original_numel},
      {"padded_numel", flat_param_shard.padded_numel},
      {"shard_begin", flat_param_shard.shard_begin},
      {"shard_end", flat_param_shard.shard_end},
      {"numel", flat_param_shard.shard.numel()},
  };
  rank_meta["parameters"] = nlohmann::json::array();
  for (size_t i = 0; i < params.size(); ++i) {
    nlohmann::json item;
    item["index"] = i;
    item["name"] = param_names[i];
    item["dtype"] = c10::toString(params[i].scalar_type());
    item["shape"] = std::vector<int64_t>(params[i].sizes().begin(), params[i].sizes().end());
    item["checkpoint_key"] = "param_" + std::to_string(i);
    rank_meta["parameters"].push_back(std::move(item));
  }
  write_json_atomic(step_dir / rank_meta_file, rank_meta);
  (void)world_size;
  (void)pp_size;
  (void)tp_size;
}

void write_checkpoint_manifest(const std::string& checkpoint_dir,
                               int64_t step,
                               int64_t world_size,
                               int64_t dp_size,
                               int64_t pp_size,
                               int64_t tp_size,
                               const cverl::torch_backend::Fp32MasterAdamW& optimizer) {
  if (checkpoint_dir.empty()) {
    return;
  }
  const auto step_dir = std::filesystem::path(checkpoint_dir) / checkpoint_step_name(step);
  nlohmann::json manifest;
  manifest["step"] = step + 1;
  manifest["format"] = "cverl_pp_tp_rank_local_checkpoint_v2";
  manifest["complete"] = true;
  manifest["use_master_weights"] = optimizer.uses_master_weights();
  manifest["world_size"] = world_size;
  manifest["data_parallel"] = dp_size;
  manifest["pipeline_parallel"] = pp_size;
  manifest["tensor_parallel"] = tp_size;
  manifest["optimizer"] = {
      {"type", "Fp32MasterAdamW"},
      {"lr", optimizer.options().lr},
      {"beta1", optimizer.options().beta1},
      {"beta2", optimizer.options().beta2},
      {"eps", optimizer.options().eps},
      {"weight_decay", optimizer.options().weight_decay},
      {"step", optimizer.step_count()},
  };
  manifest["rank_files"] = nlohmann::json::array();
  for (int64_t rank = 0; rank < world_size; ++rank) {
    const int64_t tp = rank % tp_size;
    const int64_t pp = (rank / tp_size) % pp_size;
    const int64_t dp = rank / (pp_size * tp_size);
    const std::string param_file = "rank_" + std::to_string(rank) + ".pt";
    const std::string metadata_file = "rank_" + std::to_string(rank) + ".json";
    if (!std::filesystem::exists(step_dir / param_file) ||
        !std::filesystem::exists(step_dir / metadata_file)) {
      throw std::runtime_error("checkpoint is incomplete before manifest write");
    }
    manifest["rank_files"].push_back({
        {"global_rank", rank},
        {"data_rank", dp},
        {"pipeline_rank", pp},
        {"tensor_rank", tp},
        {"param_file", param_file},
        {"metadata_file", metadata_file},
    });
  }
  write_json_atomic(step_dir / "manifest.json", manifest);
  write_text_atomic(std::filesystem::path(checkpoint_dir) / "latest_checkpoint.txt",
                    checkpoint_step_name(step) + "\n");
}

void write_flat_checkpoint_manifest(const std::string& checkpoint_dir,
                                    int64_t step,
                                    int64_t world_size,
                                    int64_t dp_size,
                                    int64_t pp_size,
                                    int64_t tp_size,
                                    const cverl::torch_backend::FlatAdamW& optimizer) {
  if (checkpoint_dir.empty()) {
    return;
  }
  const auto step_dir = std::filesystem::path(checkpoint_dir) / checkpoint_step_name(step);
  nlohmann::json manifest;
  manifest["step"] = step + 1;
  manifest["format"] = "cverl_pp_tp_rank_local_checkpoint_v3";
  manifest["complete"] = true;
  manifest["optimizer_kind"] = "FlatAdamW";
  manifest["world_size"] = world_size;
  manifest["data_parallel"] = dp_size;
  manifest["pipeline_parallel"] = pp_size;
  manifest["tensor_parallel"] = tp_size;
  manifest["optimizer"] = {
      {"type", "FlatAdamW"},
      {"lr", optimizer.options().lr},
      {"beta1", optimizer.options().beta1},
      {"beta2", optimizer.options().beta2},
      {"eps", optimizer.options().eps},
      {"weight_decay", optimizer.options().weight_decay},
      {"step", optimizer.step_count()},
  };
  manifest["rank_files"] = nlohmann::json::array();
  for (int64_t rank = 0; rank < world_size; ++rank) {
    const int64_t tp = rank % tp_size;
    const int64_t pp = (rank / tp_size) % pp_size;
    const int64_t dp = rank / (pp_size * tp_size);
    const std::string param_file = "rank_" + std::to_string(rank) + ".pt";
    const std::string metadata_file = "rank_" + std::to_string(rank) + ".json";
    if (!std::filesystem::exists(step_dir / param_file) ||
        !std::filesystem::exists(step_dir / metadata_file)) {
      throw std::runtime_error("flat checkpoint is incomplete before manifest write");
    }
    manifest["rank_files"].push_back({
        {"global_rank", rank},
        {"data_rank", dp},
        {"pipeline_rank", pp},
        {"tensor_rank", tp},
        {"param_file", param_file},
        {"metadata_file", metadata_file},
    });
  }
  write_json_atomic(step_dir / "manifest.json", manifest);
  write_text_atomic(std::filesystem::path(checkpoint_dir) / "latest_checkpoint.txt",
                    checkpoint_step_name(step) + "\n");
}

int64_t load_rank_checkpoint(const std::string& checkpoint_path,
                             int64_t global_rank,
                             int64_t dp_rank,
                             int64_t pp_rank,
                             int64_t tp_rank,
                             const std::vector<std::string>& param_names,
                             std::vector<torch::Tensor>& params,
                             cverl::torch_backend::Fp32MasterAdamW& optimizer,
                             const std::vector<int64_t>& optimizer_param_indices) {
  if (checkpoint_path.empty()) {
    return 0;
  }
  auto step_dir = std::filesystem::path(checkpoint_path);
  if (!std::filesystem::exists(step_dir / ("rank_" + std::to_string(global_rank) + ".json")) &&
      std::filesystem::exists(step_dir / "latest_checkpoint.txt")) {
    std::ifstream latest_in(step_dir / "latest_checkpoint.txt");
    std::string latest_step;
    latest_in >> latest_step;
    if (latest_step.empty()) {
      throw std::runtime_error("resume checkpoint latest_checkpoint.txt is empty");
    }
    step_dir /= latest_step;
  }
  const auto rank_meta_path = step_dir / ("rank_" + std::to_string(global_rank) + ".json");
  const auto manifest_path = step_dir / "manifest.json";
  if (!std::filesystem::exists(rank_meta_path)) {
    throw std::runtime_error("resume checkpoint missing rank metadata: " + rank_meta_path.string());
  }

  nlohmann::json rank_meta;
  {
    std::ifstream in(rank_meta_path);
    in >> rank_meta;
  }
  if (rank_meta.value("global_rank", global_rank) != global_rank ||
      rank_meta.value("data_rank", dp_rank) != dp_rank ||
      rank_meta.value("pipeline_rank", pp_rank) != pp_rank ||
      rank_meta.value("tensor_rank", tp_rank) != tp_rank) {
    throw std::runtime_error("resume checkpoint rank metadata does not match current rank topology");
  }
  const auto& param_meta = rank_meta.at("parameters");
  if (!param_meta.is_array() || param_meta.size() != param_names.size() || param_names.size() != params.size()) {
    throw std::runtime_error("resume checkpoint parameter metadata size mismatch");
  }
  for (size_t i = 0; i < param_names.size(); ++i) {
    const auto& item = param_meta.at(i);
    if (item.at("index").get<size_t>() != i || item.at("name").get<std::string>() != param_names[i]) {
      throw std::runtime_error("resume checkpoint parameter order/name mismatch at index " + std::to_string(i));
    }
    const auto shape = item.at("shape").get<std::vector<int64_t>>();
    if (shape != std::vector<int64_t>(params[i].sizes().begin(), params[i].sizes().end())) {
      throw std::runtime_error("resume checkpoint parameter shape mismatch for " + param_names[i]);
    }
  }
  std::vector<int64_t> checkpoint_optimizer_indices;
  if (rank_meta.contains("optimizer_param_indices")) {
    checkpoint_optimizer_indices = rank_meta.at("optimizer_param_indices").get<std::vector<int64_t>>();
  } else {
    checkpoint_optimizer_indices = full_index_list(params.size());
  }
  if (checkpoint_optimizer_indices != optimizer_param_indices) {
    throw std::runtime_error("resume checkpoint optimizer shard assignment mismatch");
  }

  int64_t manifest_step = rank_meta.value("optimizer_step", 0);
  if (std::filesystem::exists(manifest_path)) {
    nlohmann::json manifest;
    std::ifstream in(manifest_path);
    in >> manifest;
    if (manifest.value("format", std::string{}) != "cverl_pp_tp_rank_local_checkpoint_v2") {
      throw std::runtime_error("unsupported resume checkpoint format");
    }
    if (!manifest.value("complete", false)) {
      throw std::runtime_error("resume checkpoint manifest is not marked complete");
    }
    manifest_step = manifest.at("optimizer").at("step").get<int64_t>();
  }

  const std::string rank_file = rank_meta.value("param_file", "rank_" + std::to_string(global_rank) + ".pt");
  torch::serialize::InputArchive archive;
  archive.load_from((step_dir / rank_file).string());
  std::vector<torch::Tensor> checkpoint_params;
  std::vector<torch::Tensor> exp_avg;
  std::vector<torch::Tensor> exp_avg_sq;
  checkpoint_params.reserve(optimizer_param_indices.size());
  exp_avg.reserve(optimizer_param_indices.size());
  exp_avg_sq.reserve(optimizer_param_indices.size());
  {
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < params.size(); ++i) {
      torch::Tensor param;
      archive.read("param_" + std::to_string(i), param);
      params[i].copy_(param.detach().to(params[i].device(), params[i].scalar_type()));
    }
  }
  for (size_t i = 0; i < optimizer_param_indices.size(); ++i) {
    const int64_t param_index = optimizer_param_indices[i];
    torch::Tensor param;
    torch::Tensor m;
    torch::Tensor v;
    archive.read("param_" + std::to_string(param_index), param);
    archive.read("optim_exp_avg_" + std::to_string(i), m);
    archive.read("optim_exp_avg_sq_" + std::to_string(i), v);
    checkpoint_params.push_back(param);
    exp_avg.push_back(m);
    exp_avg_sq.push_back(v);
  }
  torch::Tensor optim_step_tensor;
  archive.read("optim_step", optim_step_tensor);
  const int64_t archive_step = optim_step_tensor.item<int64_t>();
  if (manifest_step != archive_step) {
    throw std::runtime_error("resume checkpoint optimizer step mismatch");
  }
  optimizer.load_state(checkpoint_params, exp_avg, exp_avg_sq, archive_step);
  return archive_step;
}

int64_t load_rank_flat_checkpoint(const std::string& checkpoint_path,
                                  int64_t global_rank,
                                  int64_t dp_rank,
                                  int64_t pp_rank,
                                  int64_t tp_rank,
                                  const std::vector<std::string>& param_names,
                                  std::vector<torch::Tensor>& params,
                                  const cverl::distributed::FlatParameterShard& flat_param_shard,
                                  cverl::torch_backend::FlatAdamW& optimizer) {
  if (checkpoint_path.empty()) {
    return 0;
  }
  auto step_dir = std::filesystem::path(checkpoint_path);
  if (!std::filesystem::exists(step_dir / ("rank_" + std::to_string(global_rank) + ".json")) &&
      std::filesystem::exists(step_dir / "latest_checkpoint.txt")) {
    std::ifstream latest_in(step_dir / "latest_checkpoint.txt");
    std::string latest_step;
    latest_in >> latest_step;
    if (latest_step.empty()) {
      throw std::runtime_error("resume flat checkpoint latest_checkpoint.txt is empty");
    }
    step_dir /= latest_step;
  }
  const auto rank_meta_path = step_dir / ("rank_" + std::to_string(global_rank) + ".json");
  const auto manifest_path = step_dir / "manifest.json";
  if (!std::filesystem::exists(rank_meta_path)) {
    throw std::runtime_error("resume flat checkpoint missing rank metadata: " + rank_meta_path.string());
  }

  nlohmann::json rank_meta;
  {
    std::ifstream in(rank_meta_path);
    in >> rank_meta;
  }
  if (rank_meta.value("global_rank", global_rank) != global_rank ||
      rank_meta.value("data_rank", dp_rank) != dp_rank ||
      rank_meta.value("pipeline_rank", pp_rank) != pp_rank ||
      rank_meta.value("tensor_rank", tp_rank) != tp_rank ||
      rank_meta.value("optimizer_kind", std::string{}) != "FlatAdamW") {
    throw std::runtime_error("resume flat checkpoint rank metadata does not match current rank topology");
  }
  const auto& param_meta = rank_meta.at("parameters");
  if (!param_meta.is_array() || param_meta.size() != param_names.size() || param_names.size() != params.size()) {
    throw std::runtime_error("resume flat checkpoint parameter metadata size mismatch");
  }
  for (size_t i = 0; i < param_names.size(); ++i) {
    const auto& item = param_meta.at(i);
    if (item.at("index").get<size_t>() != i || item.at("name").get<std::string>() != param_names[i]) {
      throw std::runtime_error("resume flat checkpoint parameter order/name mismatch at index " + std::to_string(i));
    }
    const auto shape = item.at("shape").get<std::vector<int64_t>>();
    if (shape != std::vector<int64_t>(params[i].sizes().begin(), params[i].sizes().end())) {
      throw std::runtime_error("resume flat checkpoint parameter shape mismatch for " + param_names[i]);
    }
  }
  const auto& shard_meta = rank_meta.at("flat_parameter_shard");
  if (shard_meta.at("original_numel").get<int64_t>() != flat_param_shard.original_numel ||
      shard_meta.at("padded_numel").get<int64_t>() != flat_param_shard.padded_numel ||
      shard_meta.at("shard_begin").get<int64_t>() != flat_param_shard.shard_begin ||
      shard_meta.at("shard_end").get<int64_t>() != flat_param_shard.shard_end ||
      shard_meta.at("numel").get<int64_t>() != flat_param_shard.shard.numel()) {
    throw std::runtime_error("resume flat checkpoint shard metadata mismatch");
  }

  int64_t manifest_step = rank_meta.value("optimizer_step", 0);
  if (std::filesystem::exists(manifest_path)) {
    nlohmann::json manifest;
    std::ifstream in(manifest_path);
    in >> manifest;
    const std::string format = manifest.value("format", std::string{});
    if (format != "cverl_pp_tp_rank_local_checkpoint_v3" ||
        manifest.value("optimizer_kind", std::string{}) != "FlatAdamW") {
      throw std::runtime_error("unsupported resume flat checkpoint format");
    }
    if (!manifest.value("complete", false)) {
      throw std::runtime_error("resume flat checkpoint manifest is not marked complete");
    }
    manifest_step = manifest.at("optimizer").at("step").get<int64_t>();
  }

  const std::string rank_file = rank_meta.value("param_file", "rank_" + std::to_string(global_rank) + ".pt");
  torch::serialize::InputArchive archive;
  archive.load_from((step_dir / rank_file).string());
  {
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < params.size(); ++i) {
      torch::Tensor param;
      archive.read("param_" + std::to_string(i), param);
      params[i].copy_(param.detach().to(params[i].device(), params[i].scalar_type()));
    }
  }
  torch::Tensor flat_param;
  torch::Tensor exp_avg;
  torch::Tensor exp_avg_sq;
  archive.read("flat_param_shard", flat_param);
  archive.read("flat_exp_avg", exp_avg);
  archive.read("flat_exp_avg_sq", exp_avg_sq);
  torch::Tensor optim_step_tensor;
  archive.read("optim_step", optim_step_tensor);
  const int64_t archive_step = optim_step_tensor.item<int64_t>();
  if (manifest_step != archive_step) {
    throw std::runtime_error("resume flat checkpoint optimizer step mismatch");
  }
  optimizer.load_state(flat_param, exp_avg, exp_avg_sq, archive_step);
  return archive_step;
}

void append_metrics_csv(const std::string& path,
                        int64_t step,
                        int64_t dp_size,
                        int64_t pp_size,
                        int64_t tp_size,
                        int64_t micro_batches,
                        int64_t layers,
                        int64_t prompt_len,
                        int64_t response_len,
                        int64_t rollout_rows,
                        double mean_reward,
                        double success_rate,
                        double adv_abs_sum,
                        double total_loss,
                        double total_kl_loss,
                        double total_kl,
                        double total_clipfrac,
                        double total_grad_norm,
                        double global_grad_norm,
                        double grad_clip_scale,
                        double total_param_delta) {
  if (path.empty()) {
    return;
  }
  const bool write_header = !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
  const auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  std::ofstream out(path, std::ios::app);
  if (write_header) {
    out << "step,dp,pp,tp,micro_batches,layers,prompt_len,response_len,rollout_rows,"
        << "mean_reward,success_rate,adv_abs_sum,loss_sum,kl_loss_sum,ppo_kl_sum,clipfrac_sum,"
        << "grad_norm_sum,global_grad_norm,grad_clip_scale,param_delta_sum\n";
  }
  out << step << ","
      << dp_size << ","
      << pp_size << ","
      << tp_size << ","
      << micro_batches << ","
      << layers << ","
      << prompt_len << ","
      << response_len << ","
      << rollout_rows << ","
      << mean_reward << ","
      << success_rate << ","
      << adv_abs_sum << ","
      << total_loss << ","
      << total_kl_loss << ","
      << total_kl << ","
      << total_clipfrac << ","
      << total_grad_norm << ","
      << global_grad_norm << ","
      << grad_clip_scale << ","
      << total_param_delta << "\n";
}

PpPpoBatch load_rollout_ppo_batch(const std::string& path,
                                  const std::string& tokenizer_json,
                                  int64_t prompt_len,
                                  int64_t response_len,
                                  int64_t expected_step,
                                  torch::Device device) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open rollout JSON: " + path);
  }
  nlohmann::json doc = nlohmann::json::parse(in);
  if (doc.contains("step")) {
    const int64_t actual_step = doc.at("step").get<int64_t>();
    if (actual_step != expected_step) {
      throw std::runtime_error(
          "rollout batch step mismatch for " + path + ": expected " +
          std::to_string(expected_step) + ", got " + std::to_string(actual_step));
    }
  }
  auto prompts = json_string_vector(doc, "prompts");
  auto answers = json_string_vector(doc, "answers");
  auto responses = json_string_vector(doc, "responses");
  auto prompt_indices = json_u32_vector(doc, "prompt_indices");
  auto sample_indices = json_u32_vector(doc, "sample_indices");
  if (responses.size() != prompt_indices.size() || responses.size() != sample_indices.size()) {
    throw std::runtime_error("rollout JSON response/index size mismatch: " + path);
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

  cverl::text::HfBpeTokenizerOptions tok_opts;
  tok_opts.tokenizer_json_path = tokenizer_json;
  cverl::text::HfBpeTokenizer tokenizer(tok_opts);
  cverl::reward::Gsm8kRewardOptions reward_opts;
  reward_opts.method = cverl::reward::Gsm8kExtractionMethod::Flexible;
  cverl::rollout::RolloutBatchOptions batch_opts;
  batch_opts.max_prompt_tokens = prompt_len;
  batch_opts.max_response_tokens = response_len;
  auto rollout_batch =
      cverl::rollout::build_gsm8k_rollout_batch(rollout, prompts, answers, reward_opts, tokenizer, batch_opts);

  auto prompt_ids = rollout_batch.prompt_ids.to(device);
  auto response_ids = rollout_batch.response_ids.to(device);
  auto response_mask = rollout_batch.response_mask.to(device);
  auto token_rewards = rollout_batch.token_rewards.to(device);
  auto group_ids = rollout_batch.group_ids.to(device);
  torch::Tensor returns;
  auto advantages = cverl::torch_backend::grpo_outcome_advantage(
      token_rewards, response_mask, group_ids, 1.0e-6, true, &returns);

  PpPpoBatch out;
  out.rows = prompt_ids.size(0);
  out.mean_reward = rollout_batch.scalar_rewards.mean().item<double>();
  out.success_rate =
      (rollout_batch.scalar_rewards >= reward_opts.correct_score - 1e-6).to(torch::kFloat32).mean().item<double>();
  out.adv_abs_sum = advantages.abs().sum().item<double>();
  out.token_ids.reserve(static_cast<size_t>(out.rows));
  out.advantages.reserve(static_cast<size_t>(out.rows));
  out.response_masks.reserve(static_cast<size_t>(out.rows));
  for (int64_t i = 0; i < out.rows; ++i) {
    out.token_ids.push_back(torch::cat({prompt_ids.index({i}), response_ids.index({i})}, 0).view({1, prompt_len + response_len}));
    out.advantages.push_back(advantages.index({i}).view({1, response_len}));
    out.response_masks.push_back(response_mask.index({i}).view({1, response_len}));
  }
  return out;
}

cverl::rollout::GpuRolloutTensorKind parse_ipc_kind(const std::string& kind) {
  if (kind == "prompt_ids") return cverl::rollout::GpuRolloutTensorKind::PromptIds;
  if (kind == "response_ids") return cverl::rollout::GpuRolloutTensorKind::ResponseIds;
  if (kind == "response_mask") return cverl::rollout::GpuRolloutTensorKind::ResponseMask;
  if (kind == "old_log_probs") return cverl::rollout::GpuRolloutTensorKind::OldLogProbs;
  if (kind == "ref_log_probs") return cverl::rollout::GpuRolloutTensorKind::RefLogProbs;
  if (kind == "rewards") return cverl::rollout::GpuRolloutTensorKind::Rewards;
  if (kind == "advantages") return cverl::rollout::GpuRolloutTensorKind::Advantages;
  if (kind == "group_ids") return cverl::rollout::GpuRolloutTensorKind::GroupIds;
  throw std::invalid_argument("unknown rollout IPC tensor kind: " + kind);
}

cverl::rollout::GpuIpcDType parse_ipc_dtype(const std::string& dtype) {
  if (dtype == "int64") return cverl::rollout::GpuIpcDType::Int64;
  if (dtype == "float32") return cverl::rollout::GpuIpcDType::Float32;
  if (dtype == "uint8") return cverl::rollout::GpuIpcDType::UInt8;
  if (dtype == "bfloat16") return cverl::rollout::GpuIpcDType::BFloat16;
  if (dtype == "float16") return cverl::rollout::GpuIpcDType::Float16;
  throw std::invalid_argument("unknown rollout IPC tensor dtype: " + dtype);
}

torch::Tensor find_ipc_tensor(const std::unordered_map<std::string, torch::Tensor>& tensors,
                              const std::string& name) {
  auto it = tensors.find(name);
  if (it == tensors.end()) {
    throw std::runtime_error("rollout IPC manifest missing tensor: " + name);
  }
  return it->second;
}

PpPpoBatch load_rollout_ipc_batch(const std::string& path,
                                  int64_t expected_prompt_len,
                                  int64_t expected_response_len,
                                  int64_t expected_step,
                                  int32_t device_index) {
#ifndef CVERL_ENABLE_CUDA
  (void)path;
  (void)expected_prompt_len;
  (void)expected_response_len;
  (void)expected_step;
  (void)device_index;
  throw std::runtime_error("rollout CUDA IPC requires CVERL_ENABLE_CUDA=ON");
#else
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open rollout IPC manifest: " + path);
  }
  nlohmann::json doc = nlohmann::json::parse(in);
  const int64_t step = doc.value("step", expected_step);
  if (step != expected_step) {
    throw std::runtime_error("rollout IPC step mismatch for " + path);
  }
  const int64_t batch = doc.at("batch").get<int64_t>();
  const int64_t prompt_len = doc.at("prompt_len").get<int64_t>();
  const int64_t response_len = doc.at("response_len").get<int64_t>();
  if (prompt_len != expected_prompt_len || response_len != expected_response_len) {
    throw std::runtime_error("rollout IPC prompt/response length mismatch for " + path);
  }

  std::unordered_map<std::string, torch::Tensor> tensors;
  for (const auto& item : doc.at("tensors")) {
    cverl::rollout::GpuIpcTensorHandle handle;
    const std::string kind_name = item.at("kind").get<std::string>();
    handle.kind = parse_ipc_kind(kind_name);
    handle.dtype = parse_ipc_dtype(item.at("dtype").get<std::string>());
    handle.device_index = item.value("device_index", device_index);
    const auto shape = item.at("shape").get<std::vector<int64_t>>();
    const auto strides = item.value("strides", std::vector<int64_t>{});
    handle.ndim = static_cast<int32_t>(shape.size());
    if (handle.ndim <= 0 || handle.ndim > 4) {
      throw std::runtime_error("invalid rollout IPC tensor rank for " + kind_name);
    }
    handle.numel = 1;
    for (int32_t i = 0; i < handle.ndim; ++i) {
      handle.sizes[static_cast<size_t>(i)] = shape[static_cast<size_t>(i)];
      handle.strides[static_cast<size_t>(i)] =
          strides.empty() ? (i + 1 == handle.ndim ? 1 : 0) : strides[static_cast<size_t>(i)];
      handle.numel *= static_cast<uint64_t>(shape[static_cast<size_t>(i)]);
    }
    if (strides.empty()) {
      int64_t stride = 1;
      for (int32_t i = handle.ndim - 1; i >= 0; --i) {
        handle.strides[static_cast<size_t>(i)] = stride;
        stride *= handle.sizes[static_cast<size_t>(i)];
      }
    }
    cverl::rollout::cuda_ipc_handle_from_hex(item.at("handle").get<std::string>(), &handle);
    tensors.emplace(kind_name, cverl::rollout::import_cuda_ipc_tensor(handle, device_index));
  }

  auto prompt_ids = find_ipc_tensor(tensors, "prompt_ids");
  auto response_ids = find_ipc_tensor(tensors, "response_ids");
  auto response_mask = find_ipc_tensor(tensors, "response_mask").to(torch::kFloat32);
  auto advantages = find_ipc_tensor(tensors, "advantages").to(torch::kFloat32);
  torch::Tensor old_log_probs;
  auto old_it = tensors.find("old_log_probs");
  if (old_it != tensors.end()) {
    old_log_probs = old_it->second.to(torch::kFloat32);
  }
  torch::Tensor ref_log_probs;
  auto ref_it = tensors.find("ref_log_probs");
  if (ref_it != tensors.end()) {
    ref_log_probs = ref_it->second.to(torch::kFloat32);
  }

  PpPpoBatch out;
  out.rows = batch;
  out.mean_reward = doc.value("mean_reward", 0.0);
  out.success_rate = doc.value("success_rate", 0.0);
  out.adv_abs_sum = advantages.abs().sum().item<double>();
  out.token_ids.reserve(static_cast<size_t>(batch));
  out.advantages.reserve(static_cast<size_t>(batch));
  out.response_masks.reserve(static_cast<size_t>(batch));
  out.old_log_probs.reserve(static_cast<size_t>(batch));
  out.ref_log_probs.reserve(static_cast<size_t>(batch));
  for (int64_t i = 0; i < batch; ++i) {
    out.token_ids.push_back(torch::cat({prompt_ids.index({i}), response_ids.index({i})}, 0).view({1, prompt_len + response_len}));
    out.advantages.push_back(advantages.index({i}).view({1, response_len}));
    out.response_masks.push_back(response_mask.index({i}).view({1, response_len}));
    if (old_log_probs.defined()) {
      out.old_log_probs.push_back(old_log_probs.index({i}).view({1, response_len}));
    }
    if (ref_log_probs.defined()) {
      out.ref_log_probs.push_back(ref_log_probs.index({i}).view({1, response_len}));
    }
  }
  return out;
#endif
}

PpPpoBatch convert_gpu_rollout_batch(const cverl::rollout::GpuRolloutBatch& gpu_batch,
                                     int64_t expected_prompt_len,
                                     int64_t expected_response_len) {
  if (!gpu_batch.prompt_ids.defined() || !gpu_batch.response_ids.defined() ||
      !gpu_batch.response_mask.defined() || !gpu_batch.advantages.defined()) {
    throw std::runtime_error("NCCL GPU rollout batch missing required tensors");
  }
  const int64_t batch = gpu_batch.prompt_ids.size(0);
  const int64_t prompt_len = gpu_batch.prompt_ids.size(1);
  const int64_t response_len = gpu_batch.response_ids.size(1);
  if (prompt_len != expected_prompt_len || response_len != expected_response_len) {
    throw std::runtime_error("NCCL GPU rollout batch prompt/response length mismatch");
  }
  PpPpoBatch out;
  out.rows = batch;
  out.adv_abs_sum = gpu_batch.advantages.to(torch::kFloat32).abs().sum().item<double>();
  if (gpu_batch.rewards.defined() && gpu_batch.rewards.numel() > 0) {
    auto rewards = gpu_batch.rewards.to(torch::kFloat32);
    out.mean_reward = rewards.mean().item<double>();
    out.success_rate = (rewards >= 1.0 - 1e-6).to(torch::kFloat32).mean().item<double>();
  }
  out.token_ids.reserve(static_cast<size_t>(batch));
  out.advantages.reserve(static_cast<size_t>(batch));
  out.response_masks.reserve(static_cast<size_t>(batch));
  out.old_log_probs.reserve(static_cast<size_t>(batch));
  out.ref_log_probs.reserve(static_cast<size_t>(batch));
  auto response_mask = gpu_batch.response_mask.to(torch::kFloat32);
  auto advantages = gpu_batch.advantages.to(torch::kFloat32);
  torch::Tensor old_log_probs = gpu_batch.old_log_probs.defined() ? gpu_batch.old_log_probs.to(torch::kFloat32) : torch::Tensor();
  torch::Tensor ref_log_probs = gpu_batch.ref_log_probs.defined() ? gpu_batch.ref_log_probs.to(torch::kFloat32) : torch::Tensor();
  for (int64_t i = 0; i < batch; ++i) {
    out.token_ids.push_back(torch::cat({gpu_batch.prompt_ids.index({i}), gpu_batch.response_ids.index({i})}, 0)
                                .view({1, prompt_len + response_len})
                                .contiguous());
    out.advantages.push_back(advantages.index({i}).view({1, response_len}).contiguous());
    out.response_masks.push_back(response_mask.index({i}).view({1, response_len}).contiguous());
    if (old_log_probs.defined()) {
      out.old_log_probs.push_back(old_log_probs.index({i}).view({1, response_len}).contiguous());
    }
    if (ref_log_probs.defined()) {
      out.ref_log_probs.push_back(ref_log_probs.index({i}).view({1, response_len}).contiguous());
    }
  }
  return out;
}

std::vector<torch::Tensor> load_jsonl_token_batches(const std::string& jsonl_path,
                                                    const std::string& tokenizer_json,
                                                    int64_t seq_len,
                                                    int64_t max_examples,
                                                    torch::Device device) {
  if (jsonl_path.empty()) {
    return {};
  }
  cverl::data::JsonlDatasetOptions data_opts;
  data_opts.path = jsonl_path;
  data_opts.prompt_field = "prompt";
  data_opts.answer_field = "answer";
  data_opts.max_examples = max_examples;
  auto examples = cverl::data::load_prompt_answer_jsonl(data_opts);
  if (examples.empty()) {
    throw std::runtime_error("jsonl input produced no examples: " + jsonl_path);
  }

  cverl::text::HfBpeTokenizerOptions tok_opts;
  tok_opts.tokenizer_json_path = tokenizer_json;
  cverl::text::HfBpeTokenizer tokenizer(tok_opts);
  const int32_t pad_id = tokenizer.pad_id() >= 0 ? tokenizer.pad_id() : 0;

  std::vector<torch::Tensor> batches;
  batches.reserve(examples.size());
  for (const auto& ex : examples) {
    cverl::text::EncodeOptions enc_opts;
    enc_opts.add_bos = false;
    enc_opts.add_eos = false;
    auto ids32 = tokenizer.encode("Question: " + ex.prompt + "\nAnswer:", enc_opts);
    std::vector<int64_t> ids(static_cast<size_t>(seq_len), static_cast<int64_t>(pad_id));
    if (ids32.empty()) {
      ids32.push_back(pad_id);
    }
    if (static_cast<int64_t>(ids32.size()) >= seq_len) {
      const int64_t start = static_cast<int64_t>(ids32.size()) - seq_len;
      for (int64_t i = 0; i < seq_len; ++i) {
        ids[static_cast<size_t>(i)] = ids32[static_cast<size_t>(start + i)];
      }
    } else {
      const int64_t pad = seq_len - static_cast<int64_t>(ids32.size());
      for (int64_t i = 0; i < static_cast<int64_t>(ids32.size()); ++i) {
        ids[static_cast<size_t>(pad + i)] = ids32[static_cast<size_t>(i)];
      }
    }
    batches.push_back(torch::tensor(ids, torch::TensorOptions().dtype(torch::kLong).device(device)).view({1, seq_len}));
  }
  return batches;
}

torch::Tensor make_token_ids(int64_t seq_len,
                             int64_t micro_batch,
                             int64_t step,
                             bool vary_tokens_by_step,
                             const PpPpoBatch* rollout_batch,
                             const std::vector<torch::Tensor>& jsonl_batches,
                             const cverl::Qwen35TextConfig& config,
                             torch::Device device) {
  if (rollout_batch != nullptr && rollout_batch->rows > 0) {
    const int64_t idx = micro_batch % rollout_batch->rows;
    return rollout_batch->token_ids[static_cast<size_t>(idx)];
  }
  if (!jsonl_batches.empty()) {
    const int64_t idx = (step * 1024 + micro_batch) % static_cast<int64_t>(jsonl_batches.size());
    return jsonl_batches[static_cast<size_t>(idx)];
  }
  auto ids = torch::arange(1, seq_len + 1, torch::TensorOptions().dtype(torch::kLong).device(device)).view({1, seq_len});
  const int64_t offset = micro_batch + (vary_tokens_by_step ? step : 0);
  return (ids + offset).remainder(config.vocab_size - 1).clamp_min(1);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      throw std::runtime_error(
          "usage: qwen3_5_pp_tp_ppo_trainer MODEL_DIR --global-rank R --device D --layers N");
    }

    const std::string model_dir = argv[1];
    const int64_t global_rank =
        arg_i64(argc, argv, "--global-rank", std::getenv("RANK") ? std::stoll(std::getenv("RANK")) : 0);
    const int64_t dp_size = arg_i64(argc, argv, "--dp-size", 1);
    const int64_t pp_size = arg_i64(argc, argv, "--pp-size", 2);
    const int64_t tp_size = arg_i64(argc, argv, "--tp-size", 2);
    const int64_t world_size = dp_size * pp_size * tp_size;
    const int64_t device_idx =
        arg_i64(argc, argv, "--device", std::getenv("LOCAL_RANK") ? std::stoll(std::getenv("LOCAL_RANK")) : global_rank);
    const int64_t layers = arg_i64(argc, argv, "--layers", 2);
    const int64_t prompt_len = arg_i64(argc, argv, "--prompt-len", 2);
    const int64_t response_len = arg_i64(argc, argv, "--response-len", 2);
    const int64_t micro_batches = arg_i64(argc, argv, "--micro-batches", pp_size);
    const int64_t steps = arg_i64(argc, argv, "--steps", 1);
    const double lr = arg_f64(argc, argv, "--lr", 1.0e-8);
    const double clip_ratio = arg_f64(argc, argv, "--clip-ratio", 0.2);
    const double kl_coef = arg_f64(argc, argv, "--kl-coef", 0.0);
    const auto kl_penalty_mode = parse_kl_penalty_mode(arg_str(argc, argv, "--kl-penalty", "k1"));
    const double max_grad_norm = arg_f64(argc, argv, "--max-grad-norm", 1.0);
    const int64_t dp_grad_bucket_mb = arg_i64(argc, argv, "--dp-grad-bucket-mb", 25);
    const int64_t tp_grad_bucket_mb = arg_i64(argc, argv, "--tp-grad-bucket-mb", dp_grad_bucket_mb);
    const double advantage_scale = arg_f64(argc, argv, "--advantage-scale", 1.0);
    const bool use_master_weights = arg_bool(argc, argv, "--master-weights", false);
    const bool dp_shard_optimizer = arg_bool(argc, argv, "--dp-shard-optimizer", false);
    const bool dp_flat_shard_optimizer =
        dp_shard_optimizer && arg_bool(argc, argv, "--dp-flat-shard-optimizer", true);
    const bool skip_optimizer_step = arg_bool(argc, argv, "--skip-optimizer-step", false);
    const bool vary_tokens_by_step = arg_bool(argc, argv, "--vary-tokens-by-step", false);
    const std::string jsonl_input = arg_str(argc, argv, "--jsonl-input", "");
    const std::string rollout_json = arg_str(argc, argv, "--rollout-json", "");
    const std::string rollout_dir = arg_str(argc, argv, "--rollout-dir", "");
    const std::string rollout_ipc_json = arg_str(argc, argv, "--rollout-ipc-json", "");
    const std::string rollout_ipc_dir = arg_str(argc, argv, "--rollout-ipc-dir", "");
    const std::string rollout_nccl_id_file = arg_str(argc, argv, "--rollout-nccl-id-file", "");
    const int64_t rollout_nccl_world_size = arg_i64(argc, argv, "--rollout-nccl-world-size", 0);
    const int64_t rollout_nccl_rank_offset = arg_i64(argc, argv, "--rollout-nccl-rank-offset", 0);
    const int64_t rollout_nccl_source_rank = arg_i64(argc, argv, "--rollout-nccl-source-rank", 0);
    const std::string tokenizer_json =
        arg_str(argc, argv, "--tokenizer-json", (std::filesystem::path(model_dir) / "tokenizer.json").string());
    const int64_t jsonl_max_examples = arg_i64(argc, argv, "--jsonl-max-examples", 16);
    const std::string checkpoint_dir = arg_str(argc, argv, "--checkpoint-dir", "");
    const int64_t checkpoint_every = arg_i64(argc, argv, "--checkpoint-every", 0);
    const std::string resume_checkpoint = arg_str(argc, argv, "--resume-checkpoint", "");
    const std::string metrics_csv = arg_str(argc, argv, "--metrics-csv", "");
    const auto dtype = parse_dtype(arg_str(argc, argv, "--dtype", "bfloat16"));
    const std::string id_prefix = arg_str(argc, argv, "--id-prefix", "/tmp/cverl_qwen_pp_tp_ppo");

    if (dp_size <= 0 || pp_size <= 0 || tp_size <= 0 || micro_batches <= 0 || steps <= 0) {
      throw std::invalid_argument("dp-size, pp-size, tp-size, micro-batches, and steps must be positive");
    }
    if (prompt_len <= 0 || response_len <= 0 || max_grad_norm < 0.0) {
      throw std::invalid_argument("prompt-len and response-len must be positive");
    }
    if (kl_coef < 0.0) {
      throw std::invalid_argument("--kl-coef must be non-negative");
    }
    if (dp_grad_bucket_mb <= 0) {
      throw std::invalid_argument("--dp-grad-bucket-mb must be positive");
    }
    if (tp_grad_bucket_mb <= 0) {
      throw std::invalid_argument("--tp-grad-bucket-mb must be positive");
    }
    const int64_t dp_grad_bucket_bytes = dp_grad_bucket_mb * 1024 * 1024;
    const int64_t tp_grad_bucket_bytes = tp_grad_bucket_mb * 1024 * 1024;

    const int64_t seq_len = prompt_len + response_len;
    const auto device = torch::Device(torch::kCUDA, static_cast<int>(device_idx));

    cverl::distributed::ParallelDims dims;
    dims.data_parallel = dp_size;
    dims.pipeline_parallel = pp_size;
    dims.context_parallel = 1;
    dims.tensor_parallel = tp_size;
    dims.micro_batches = micro_batches;
    cverl::distributed::ClusterSpec spec;
    spec.world_size = world_size;
    spec.rank = global_rank;
    spec.local_world_size = world_size;
    spec.local_rank = device_idx;
    spec.gpus_per_node = world_size;
    spec.parallel = dims;
    cverl::distributed::Topology topology(spec);
    auto info = topology.local_rank_info();
    auto range = topology.pipeline_layer_range(layers, info.pipeline_rank);
    auto schedule = cverl::distributed::pipeline_1f1b_schedule(topology, info);
    auto peers = cverl::distributed::pipeline_peers(topology, info);

    const std::string full_id_path = id_prefix + "_full.bin";
    const std::string tp_id_path = id_prefix + "_tp_dp" + std::to_string(info.data_rank) +
                                   "_pp" + std::to_string(info.pipeline_rank) + ".bin";
    const std::string dp_id_path = id_prefix + "_dp_pp" + std::to_string(info.pipeline_rank) +
                                   "_tp" + std::to_string(info.tensor_rank) + ".bin";
    const std::string model_id_path = id_prefix + "_model_dp" + std::to_string(info.data_rank) + ".bin";
    if (global_rank == 0) {
      cverl::distributed::write_nccl_unique_id_file(full_id_path, cverl::distributed::create_nccl_unique_id());
    }
    if (info.tensor_rank == 0) {
      cverl::distributed::write_nccl_unique_id_file(tp_id_path, cverl::distributed::create_nccl_unique_id());
    }
    if (info.data_rank == 0) {
      cverl::distributed::write_nccl_unique_id_file(dp_id_path, cverl::distributed::create_nccl_unique_id());
    }
    if (info.pipeline_rank == 0 && info.tensor_rank == 0) {
      cverl::distributed::write_nccl_unique_id_file(model_id_path, cverl::distributed::create_nccl_unique_id());
    }

    auto full_id = cverl::distributed::read_nccl_unique_id_file(full_id_path);
    auto tp_id = cverl::distributed::read_nccl_unique_id_file(tp_id_path);
    auto dp_id = cverl::distributed::read_nccl_unique_id_file(dp_id_path);
    auto model_id = cverl::distributed::read_nccl_unique_id_file(model_id_path);
    cverl::distributed::NcclCollectives full_comm(global_rank, world_size, static_cast<int>(device_idx), full_id);
    cverl::distributed::NcclCollectives tp_comm(info.tensor_rank, tp_size, static_cast<int>(device_idx), tp_id);
    cverl::distributed::NcclCollectives dp_comm(info.data_rank, dp_size, static_cast<int>(device_idx), dp_id);
    const int64_t model_parallel_size = pp_size * tp_size;
    const int64_t model_parallel_rank = info.pipeline_rank * tp_size + info.tensor_rank;
    cverl::distributed::NcclCollectives model_comm(
        model_parallel_rank, model_parallel_size, static_cast<int>(device_idx), model_id);
    cverl::distributed::ParallelGroup tp_group{info.tensor_rank, tp_size, full_group(tp_size), &tp_comm};
    cverl::distributed::ParallelGroup dp_group{info.data_rank, dp_size, full_group(dp_size), &dp_comm};
    std::unique_ptr<cverl::distributed::NcclCollectives> rollout_data_comm;
    if (!rollout_nccl_id_file.empty()) {
      if (rollout_nccl_world_size <= world_size) {
        throw std::invalid_argument("rollout NCCL world must include rollout ranks plus trainer ranks");
      }
      const int64_t rollout_data_rank = rollout_nccl_rank_offset + global_rank;
      if (rollout_data_rank < 0 || rollout_data_rank >= rollout_nccl_world_size) {
        throw std::invalid_argument("rollout NCCL data rank out of range");
      }
      auto rollout_id = cverl::distributed::read_nccl_unique_id_file(rollout_nccl_id_file);
      rollout_data_comm = std::make_unique<cverl::distributed::NcclCollectives>(
          rollout_data_rank, rollout_nccl_world_size, static_cast<int>(device_idx), rollout_id);
    }

    cverl::HfModelLoader loader(model_dir);
    cverl::Qwen35TextModel model(std::move(loader));
    model.to(device);
    const auto jsonl_batches =
        load_jsonl_token_batches(jsonl_input, tokenizer_json, seq_len, jsonl_max_examples, device);
    std::vector<torch::Tensor> params;
    std::vector<std::string> param_names;
    const auto rank_weight_names = stage_required_weight_names(
        model, layers, range.begin, range.end, peers.is_first_stage, peers.is_last_stage);
    for (const auto& name : rank_weight_names) {
      auto full_tensor = model.loader().load_tensor(name).to(dtype);
      auto tensor = qwen_megatron_tp_shard(name, full_tensor, model.config(), info.tensor_rank, tp_size)
                        .to(torch::TensorOptions().device(device).dtype(dtype))
                        .contiguous();
      tensor.set_requires_grad(true);
      model.set_weight_override(name, tensor);
      params.push_back(tensor);
      param_names.push_back(name);
    }
    std::vector<int64_t> param_bytes;
    param_bytes.reserve(params.size());
    for (const auto& p : params) {
      param_bytes.push_back(p.numel() * static_cast<int64_t>(p.element_size()));
    }
    std::vector<int64_t> dp_optimizer_owner_by_param(params.size(), 0);
    if (dp_shard_optimizer) {
      dp_optimizer_owner_by_param = cverl::distributed::greedy_parameter_owner_by_size(param_bytes, dp_size);
    }
    const auto optimizer_param_indices =
        (dp_shard_optimizer && !dp_flat_shard_optimizer)
            ? cverl::distributed::owned_parameter_indices(dp_optimizer_owner_by_param, info.data_rank)
            : full_index_list(params.size());
    auto optimizer_params =
        dp_flat_shard_optimizer ? std::vector<torch::Tensor>{} : select_params(params, optimizer_param_indices);
    cverl::torch_backend::Fp32MasterAdamWOptions optim_options;
    optim_options.lr = lr;
    optim_options.beta1 = 0.9;
    optim_options.beta2 = 0.95;
    optim_options.eps = 1.0e-8;
    optim_options.weight_decay = 0.01;
    optim_options.use_master_weights = use_master_weights;
    cverl::torch_backend::Fp32MasterAdamW optimizer(optimizer_params, optim_options);
    cverl::distributed::FlatParameterShard flat_param_shard;
    std::unique_ptr<cverl::torch_backend::FlatAdamW> flat_optimizer;
    if (dp_flat_shard_optimizer) {
      flat_param_shard = cverl::distributed::flatten_parameter_shard(params, dp_size, info.data_rank);
      flat_optimizer = std::make_unique<cverl::torch_backend::FlatAdamW>(flat_param_shard.shard, optim_options);
    }
    int64_t start_step = 0;
    if (!resume_checkpoint.empty()) {
      if (dp_flat_shard_optimizer) {
        start_step = load_rank_flat_checkpoint(resume_checkpoint,
                                               global_rank,
                                               info.data_rank,
                                               info.pipeline_rank,
                                               info.tensor_rank,
                                               param_names,
                                               params,
                                               flat_param_shard,
                                               *flat_optimizer);
        flat_param_shard.shard.copy_(flat_optimizer->parameter_shard());
      } else {
        start_step = load_rank_checkpoint(resume_checkpoint,
                                          global_rank,
                                          info.data_rank,
                                          info.pipeline_rank,
                                          info.tensor_rank,
                                          param_names,
                                          params,
                                          optimizer,
                                          optimizer_param_indices);
      }
      if (start_step > steps) {
        throw std::runtime_error("resume checkpoint step is greater than requested --steps");
      }
    }

    const std::vector<int64_t> hidden_shape{1, seq_len, model.config().hidden_size};
    const auto full_ranks = full_group(world_size);
    bool trainer_ok = true;

    for (int64_t step = start_step; step < steps; ++step) {
      zero_model_gradients(params);
      if (!dp_flat_shard_optimizer) {
        optimizer.zero_grad();
      }
      auto param_before = clone_params(params);
      auto optimizer_param_before =
          (!dp_shard_optimizer && use_master_weights)
              ? cverl::torch_backend::clone_detached(optimizer.master_parameters())
              : std::vector<torch::Tensor>{};
      torch::Tensor flat_gradient_shard;
      std::unordered_map<int64_t, torch::Tensor> stage_inputs;
      std::unordered_map<int64_t, torch::Tensor> stage_outputs;
      double loss_sum = 0.0;
      double kl_loss_sum = 0.0;
      double kl_sum = 0.0;
      double clipfrac_sum = 0.0;
      double mean_reward = 0.0;
      double success_rate = 0.0;
      double adv_abs_sum = 0.0;
      PpPpoBatch rollout_batch;
      const std::string rollout_path = rollout_json_path_for_step(rollout_json, rollout_dir, step);
      const std::string rollout_ipc_path = rollout_ipc_path_for_step(rollout_ipc_json, rollout_ipc_dir, step);
      const PpPpoBatch* active_rollout = nullptr;
      if (rollout_data_comm != nullptr) {
        if (peers.is_first_stage || peers.is_last_stage) {
          cverl::rollout::NCCLGpuBatchReceiver receiver(*rollout_data_comm, static_cast<int>(device_idx));
          auto gpu_batch = receiver.recv(rollout_nccl_source_rank);
          if (gpu_batch.descriptor.step != step + 1) {
            throw std::runtime_error("rollout NCCL batch step mismatch");
          }
          rollout_batch = convert_gpu_rollout_batch(gpu_batch, prompt_len, response_len);
          active_rollout = &rollout_batch;
          mean_reward = rollout_batch.mean_reward;
          success_rate = rollout_batch.success_rate;
          adv_abs_sum = rollout_batch.adv_abs_sum;
        }
        rollout_data_comm->barrier();
      } else if (!rollout_ipc_path.empty()) {
        rollout_batch = load_rollout_ipc_batch(
            rollout_ipc_path, prompt_len, response_len, step + 1, static_cast<int32_t>(device_idx));
        active_rollout = &rollout_batch;
        mean_reward = rollout_batch.mean_reward;
        success_rate = rollout_batch.success_rate;
        adv_abs_sum = rollout_batch.adv_abs_sum;
      } else if (!rollout_path.empty()) {
        rollout_batch = load_rollout_ppo_batch(rollout_path, tokenizer_json, prompt_len, response_len, step + 1, device);
        active_rollout = &rollout_batch;
        mean_reward = rollout_batch.mean_reward;
        success_rate = rollout_batch.success_rate;
        adv_abs_sum = rollout_batch.adv_abs_sum;
      }

      for (const auto& action : schedule) {
        if (action.op == cverl::distributed::PipelineScheduleOp::Forward) {
          torch::Tensor input;
          if (peers.is_first_stage) {
            auto ids = make_token_ids(
                seq_len, action.micro_batch, step, vary_tokens_by_step, active_rollout, jsonl_batches, model.config(), device);
            input = model.token_embeddings_tensor_parallel(ids, tp_group);
          } else {
            auto like = torch::empty(hidden_shape, torch::TensorOptions().device(device).dtype(dtype));
            input = full_comm.recv_like(like, peers.previous_rank).detach().set_requires_grad(true);
            stage_inputs[action.micro_batch] = input;
          }
          auto output = model.forward_hidden_range_tensor_parallel(
              input, range.begin, range.end, tp_group, peers.is_last_stage);
          stage_outputs[action.micro_batch] = output;
          if (!peers.is_last_stage) {
            full_comm.send(output.contiguous(), peers.next_rank);
          }
        } else {
          auto out_it = stage_outputs.find(action.micro_batch);
          if (out_it == stage_outputs.end()) {
            throw std::runtime_error("missing stage activation for backward");
          }
          if (peers.is_last_stage) {
            auto ids = make_token_ids(
                seq_len, action.micro_batch, step, vary_tokens_by_step, active_rollout, jsonl_batches, model.config(), device);
            auto response_ids = ids.slice(1, prompt_len, prompt_len + response_len);
            auto response_hidden = out_it->second.slice(1, prompt_len - 1, prompt_len + response_len - 1);
            auto log_probs = model.response_log_probs_tensor_parallel(response_hidden, response_ids, tp_group);
            torch::Tensor old_log_probs = log_probs.detach();
            torch::Tensor advantages;
            torch::Tensor response_mask;
            if (active_rollout != nullptr) {
              const int64_t idx = action.micro_batch % active_rollout->rows;
              advantages = active_rollout->advantages[static_cast<size_t>(idx)];
              response_mask = active_rollout->response_masks[static_cast<size_t>(idx)];
              if (!active_rollout->old_log_probs.empty()) {
                old_log_probs = active_rollout->old_log_probs[static_cast<size_t>(idx)];
              }
            } else {
              advantages = torch::ones_like(log_probs) * advantage_scale;
              response_mask = torch::ones_like(log_probs);
            }
            auto loss = cverl::torch_backend::ppo_clipped_loss(
                old_log_probs,
                log_probs,
                advantages,
                response_mask,
                clip_ratio,
                -1.0,
                -1.0,
                3.0,
                CVERL_LOSS_AGG_TOKEN_MEAN);
            torch::Tensor total_loss = loss.pg_loss;
            torch::Tensor kl_loss = torch::zeros({}, total_loss.options());
            if (kl_coef > 0.0) {
              if (active_rollout == nullptr || active_rollout->ref_log_probs.empty()) {
                throw std::runtime_error("--kl-coef > 0 requires ref_log_probs in rollout batch");
              }
              const int64_t idx = action.micro_batch % active_rollout->rows;
              auto ref_log_probs = active_rollout->ref_log_probs[static_cast<size_t>(idx)];
              auto kl_token = cverl::torch_backend::kl_penalty(log_probs, ref_log_probs, kl_penalty_mode);
              kl_loss = cverl::torch_backend::masked_mean(kl_token, response_mask);
              total_loss = total_loss + kl_coef * kl_loss;
            }
            auto scaled_loss = total_loss / static_cast<double>(micro_batches);
            loss_sum += scaled_loss.item<double>();
            kl_loss_sum += kl_loss.item<double>();
            kl_sum += loss.ppo_kl.item<double>();
            clipfrac_sum += loss.pg_clipfrac.item<double>();
            scaled_loss.backward();
          } else {
            auto grad = full_comm.recv_like(out_it->second.contiguous(), peers.next_rank);
            out_it->second.backward(grad);
          }

          if (!peers.is_first_stage) {
            auto in_it = stage_inputs.find(action.micro_batch);
            if (in_it == stage_inputs.end()) {
              throw std::runtime_error("missing stage input for backward");
            }
            full_comm.send(in_it->second.grad().contiguous(), peers.previous_rank);
            stage_inputs.erase(in_it);
          }
          stage_outputs.erase(out_it);
        }
      }

      sync_tp_replicated_gradients(param_names, params, tp_group, tp_grad_bucket_bytes);
      bool flat_step_done = false;
      double local_grad_norm_sq = 0.0;
      double global_grad_norm = 0.0;
      double grad_clip_scale = 1.0;
      double local_grad_norm = 0.0;
      if (dp_flat_shard_optimizer && !skip_optimizer_step) {
        auto flat_step = cverl::distributed::flat_sharded_adamw_step(params,
                                                                     flat_param_shard,
                                                                     *flat_optimizer,
                                                                     dp_comm,
                                                                     dp_group.ranks,
                                                                     full_comm,
                                                                     full_ranks,
                                                                     max_grad_norm,
                                                                     true,
                                                                     false,
                                                                     true);
        flat_gradient_shard = flat_step.gradient_shard.shard;
        local_grad_norm_sq = flat_step.local_grad_norm_sq;
        global_grad_norm = flat_step.global_grad_norm;
        grad_clip_scale = flat_step.grad_clip_scale;
        local_grad_norm = flat_step.local_grad_norm;
        flat_step_done = true;
      } else if (dp_flat_shard_optimizer) {
        auto flat_grad = cverl::distributed::reduce_scatter_flat_gradient_shard(
            params, dp_comm, dp_group.ranks, true, false);
        if (flat_grad.original_numel != flat_param_shard.original_numel ||
            flat_grad.padded_numel != flat_param_shard.padded_numel ||
            flat_grad.shard.numel() != flat_param_shard.shard.numel()) {
          throw std::runtime_error("flat DP optimizer parameter/gradient shard metadata mismatch");
        }
        flat_gradient_shard = flat_grad.shard;
      } else {
        cverl::distributed::data_parallel_sync_gradients(
            params, dp_comm, dp_group.ranks, true, dp_grad_bucket_bytes);
      }
      if (!flat_step_done) {
        local_grad_norm_sq =
            dp_flat_shard_optimizer
                ? flat_gradient_shard.pow(2).sum().item<double>()
                : (dp_shard_optimizer ? grad_l2_norm_sq(params) : optimizer.grad_l2_norm_sq());
        auto grad_sq_tensor =
            torch::tensor({local_grad_norm_sq}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        auto global_grad_sq_tensor =
            dp_flat_shard_optimizer
                ? full_comm.all_reduce(grad_sq_tensor.contiguous(), cverl::distributed::ReduceOp::Sum, full_ranks)
                : model_comm.all_reduce(
                      grad_sq_tensor.contiguous(), cverl::distributed::ReduceOp::Sum, full_group(model_parallel_size));
        global_grad_norm = std::sqrt(static_cast<double>(global_grad_sq_tensor.cpu()[0].item<float>()));
        if (max_grad_norm > 0.0 && std::isfinite(global_grad_norm) && global_grad_norm > max_grad_norm) {
          grad_clip_scale = max_grad_norm / (global_grad_norm + 1.0e-6);
          if (dp_flat_shard_optimizer) {
            flat_gradient_shard.mul_(grad_clip_scale);
          } else if (dp_shard_optimizer) {
            scale_model_gradients(params, grad_clip_scale);
          } else {
            optimizer.scale_gradients(grad_clip_scale);
          }
        }
        local_grad_norm =
            dp_flat_shard_optimizer
                ? flat_gradient_shard.norm().item<double>()
                : (dp_shard_optimizer ? grad_norm_sum(params) : optimizer.grad_norm_sum());
      }
      if (!skip_optimizer_step) {
        if (!dp_flat_shard_optimizer) {
          optimizer.step();
        }
        if (dp_shard_optimizer && !dp_flat_shard_optimizer) {
          broadcast_dp_sharded_optimizer_parameters(param_names,
                                                    params,
                                                    dp_optimizer_owner_by_param,
                                                    dp_comm,
                                                    dp_group.ranks,
                                                    dp_grad_bucket_bytes);
        }
      }
      const double local_param_delta =
          skip_optimizer_step
              ? 0.0
              : ((!dp_shard_optimizer && use_master_weights)
                     ? cverl::torch_backend::parameter_delta_sum(
                           optimizer_param_before, optimizer.master_parameters())
                     : cverl::torch_backend::parameter_delta_sum(param_before, params));
      if (checkpoint_every > 0 && (step + 1) % checkpoint_every == 0) {
        if (dp_flat_shard_optimizer) {
          save_rank_flat_checkpoint(checkpoint_dir,
                                    step,
                                    global_rank,
                                    world_size,
                                    info.data_rank,
                                    info.pipeline_rank,
                                    pp_size,
                                    info.tensor_rank,
                                    tp_size,
                                    range.begin,
                                    range.end,
                                    param_names,
                                    params,
                                    flat_param_shard,
                                    *flat_optimizer);
        } else {
          save_rank_checkpoint(checkpoint_dir,
                               step,
                               global_rank,
                               world_size,
                               info.data_rank,
                               info.pipeline_rank,
                               pp_size,
                               info.tensor_rank,
                               tp_size,
                               range.begin,
                               range.end,
                               param_names,
                               params,
                               optimizer,
                               optimizer_param_indices);
        }
        full_comm.barrier();
        if (global_rank == 0) {
          if (dp_flat_shard_optimizer) {
            write_flat_checkpoint_manifest(checkpoint_dir, step, world_size, dp_size, pp_size, tp_size, *flat_optimizer);
          } else {
            write_checkpoint_manifest(checkpoint_dir, step, world_size, dp_size, pp_size, tp_size, optimizer);
          }
        }
        full_comm.barrier();
      }

      auto grad_tensor = torch::tensor({local_grad_norm}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
      auto delta_tensor = torch::tensor({local_param_delta}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
      auto loss_tensor = torch::tensor({loss_sum}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
      auto kl_loss_tensor = torch::tensor({kl_loss_sum}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
      auto kl_tensor = torch::tensor({kl_sum}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
      auto clip_tensor = torch::tensor({clipfrac_sum}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
      auto global_grad_tensor =
          torch::tensor({global_grad_norm}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
      auto grad_clip_tensor =
          torch::tensor({grad_clip_scale}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
      auto grad_norms = full_comm.all_gather(grad_tensor.contiguous(), full_ranks, 0).cpu();
      auto param_deltas = full_comm.all_gather(delta_tensor.contiguous(), full_ranks, 0).cpu();
      auto loss_sums = full_comm.all_gather(loss_tensor.contiguous(), full_ranks, 0).cpu();
      auto kl_loss_sums = full_comm.all_gather(kl_loss_tensor.contiguous(), full_ranks, 0).cpu();
      auto kl_sums = full_comm.all_gather(kl_tensor.contiguous(), full_ranks, 0).cpu();
      auto clip_sums = full_comm.all_gather(clip_tensor.contiguous(), full_ranks, 0).cpu();
      auto global_grad_norms = full_comm.all_gather(global_grad_tensor.contiguous(), full_ranks, 0).cpu();
      auto grad_clip_scales = full_comm.all_gather(grad_clip_tensor.contiguous(), full_ranks, 0).cpu();
      full_comm.barrier();

      if (global_rank == 0) {
        bool all_have_grad = true;
        bool all_updated = true;
        bool any_updated = false;
        double total_loss = 0.0;
        double total_kl_loss = 0.0;
        double total_kl = 0.0;
        double total_clipfrac = 0.0;
        double total_grad_norm = 0.0;
        double total_param_delta = 0.0;
        double reported_global_grad_norm = 0.0;
        double reported_grad_clip_scale = 1.0;
        for (int64_t i = 0; i < grad_norms.numel(); ++i) {
          const double grad_value = grad_norms[i].item<double>();
          const double delta_value = param_deltas[i].item<double>();
          all_have_grad = all_have_grad && std::isfinite(grad_value) && grad_value > 0.0;
          all_updated = all_updated && std::isfinite(delta_value) && delta_value > 0.0;
          any_updated = any_updated || (std::isfinite(delta_value) && delta_value > 0.0);
          total_grad_norm += grad_value;
          total_param_delta += delta_value;
          total_loss += loss_sums[i].item<double>();
          total_kl_loss += kl_loss_sums[i].item<double>();
          total_kl += kl_sums[i].item<double>();
          total_clipfrac += clip_sums[i].item<double>();
          if (i == 0) {
            reported_global_grad_norm = global_grad_norms[i].item<double>();
            reported_grad_clip_scale = grad_clip_scales[i].item<double>();
          }
        }
        append_metrics_csv(metrics_csv,
                           step,
                           dp_size,
                           pp_size,
                           tp_size,
                           micro_batches,
                           layers,
                           prompt_len,
                           response_len,
                           active_rollout != nullptr ? active_rollout->rows : 0,
                           mean_reward,
                           success_rate,
                           adv_abs_sum,
                           total_loss,
                           total_kl_loss,
                           total_kl,
                           total_clipfrac,
                           total_grad_norm,
                           reported_global_grad_norm,
                           reported_grad_clip_scale,
                           total_param_delta);
        std::cout << "step=" << step
                  << " dp=" << dp_size
                  << " pp=" << pp_size
                  << " tp=" << tp_size
                  << " micro_batches=" << micro_batches
                  << " layers=" << layers
                  << " prompt_len=" << prompt_len
                  << " response_len=" << response_len
                  << " master_weights=" << (use_master_weights ? "true" : "false")
                  << " dp_shard_optimizer=" << (dp_shard_optimizer ? "true" : "false")
                  << " optimizer_params=" << optimizer_param_indices.size()
                  << " total_params=" << params.size()
                  << " dp_grad_bucket_mb=" << dp_grad_bucket_mb
                  << " tp_grad_bucket_mb=" << tp_grad_bucket_mb
                  << " resumed_from_step=" << start_step
                  << " skip_optimizer_step=" << (skip_optimizer_step ? "true" : "false")
                  << " vary_tokens_by_step=" << (vary_tokens_by_step ? "true" : "false")
                  << " jsonl_examples=" << jsonl_batches.size()
                  << " rollout_rows=" << (active_rollout != nullptr ? active_rollout->rows : 0)
                  << " kl_coef=" << kl_coef
                  << " mean_reward=" << mean_reward
                  << " success_rate=" << success_rate
                  << " adv_abs_sum=" << adv_abs_sum
                  << " loss_sum=" << total_loss
                  << " kl_loss_sum=" << total_kl_loss
                  << " ppo_kl_sum=" << total_kl
                  << " clipfrac_sum=" << total_clipfrac
                  << " grad_norm_sum=" << total_grad_norm
                  << " global_grad_norm=" << reported_global_grad_norm
                  << " grad_clip_scale=" << reported_grad_clip_scale
                  << " param_delta_sum=" << total_param_delta
                  << " all_ranks_have_grad=" << (all_have_grad ? "true" : "false")
                  << " all_ranks_updated=" << (all_updated ? "true" : "false")
                  << " any_rank_updated=" << (any_updated ? "true" : "false")
                  << "\n";
        trainer_ok = trainer_ok && all_have_grad && (skip_optimizer_step || any_updated);
      }
    }

    full_comm.barrier();
    if (global_rank == 0) {
      std::filesystem::remove(full_id_path);
      for (int64_t d = 0; d < dp_size; ++d) {
        std::filesystem::remove(id_prefix + "_model_dp" + std::to_string(d) + ".bin");
        for (int64_t pp = 0; pp < pp_size; ++pp) {
          std::filesystem::remove(
              id_prefix + "_tp_dp" + std::to_string(d) + "_pp" + std::to_string(pp) + ".bin");
          for (int64_t tp = 0; tp < tp_size; ++tp) {
            std::filesystem::remove(
                id_prefix + "_dp_pp" + std::to_string(pp) + "_tp" + std::to_string(tp) + ".bin");
          }
        }
      }
      return trainer_ok ? 0 : 2;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "qwen3_5_pp_tp_ppo_trainer failed: " << e.what() << "\n";
    return 1;
  }
}
