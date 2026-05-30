#include "cverl/model/qwen3_5_text.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <regex>
#include <stdexcept>

#include <torch/nn/functional/conv.h>
#include <torch/nn/functional/embedding.h>

namespace cverl {
namespace {

using torch::indexing::None;
using torch::indexing::Slice;

std::string read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open " + path.string());
  }
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

int64_t regex_i64(const std::string& text, const std::string& key, int64_t fallback) {
  std::smatch match;
  std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9]+)");
  if (std::regex_search(text, match, pattern)) {
    return std::stoll(match[1].str());
  }
  return fallback;
}

double regex_f64(const std::string& text, const std::string& key, double fallback) {
  std::smatch match;
  std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9]+(?:\\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)");
  if (std::regex_search(text, match, pattern)) {
    return std::stod(match[1].str());
  }
  return fallback;
}

std::vector<std::string> parse_layer_types(const std::string& text) {
  std::vector<std::string> out;
  std::smatch match;
  if (!std::regex_search(text, match, std::regex("\"layer_types\"\\s*:\\s*\\[([^\\]]+)\\]"))) {
    return out;
  }
  std::string body = match[1].str();
  std::regex item("\"([^\"]+)\"");
  for (std::sregex_iterator it(body.begin(), body.end(), item), end; it != end; ++it) {
    out.push_back((*it)[1].str());
  }
  return out;
}

torch::Tensor dense(const torch::Tensor& x, const torch::Tensor& w) {
  auto input = x.scalar_type() == w.scalar_type() ? x : x.to(w.scalar_type());
  return torch::matmul(input, w.transpose(0, 1));
}

torch::Tensor rotate_half(const torch::Tensor& x) {
  int64_t half = x.size(-1) / 2;
  auto x1 = x.index({Slice(), Slice(), Slice(), Slice(0, half)});
  auto x2 = x.index({Slice(), Slice(), Slice(), Slice(half, None)});
  return torch::cat({-x2, x1}, -1);
}

torch::Tensor l2norm(const torch::Tensor& x) {
  return x * torch::rsqrt((x * x).sum(-1, true) + 1e-6);
}

torch::Tensor repeat_kv(const torch::Tensor& x, int64_t n_rep) {
  if (n_rep == 1) {
    return x;
  }
  int64_t b = x.size(0);
  int64_t h = x.size(1);
  int64_t s = x.size(2);
  int64_t d = x.size(3);
  return x.index({Slice(), Slice(), None, Slice(), Slice()}).expand({b, h, n_rep, s, d}).reshape({b, h * n_rep, s, d});
}

std::string layer_prefix(int64_t layer_idx) {
  return "model.language_model.layers." + std::to_string(layer_idx) + ".";
}

}  // namespace

Qwen35TextConfig load_qwen35_text_config(const std::string& model_dir) {
  std::string text = read_file(std::filesystem::path(model_dir) / "config.json");
  Qwen35TextConfig cfg;
  cfg.vocab_size = regex_i64(text, "vocab_size", cfg.vocab_size);
  cfg.hidden_size = regex_i64(text, "hidden_size", cfg.hidden_size);
  cfg.num_hidden_layers = regex_i64(text, "num_hidden_layers", cfg.num_hidden_layers);
  cfg.num_attention_heads = regex_i64(text, "num_attention_heads", cfg.num_attention_heads);
  cfg.num_key_value_heads = regex_i64(text, "num_key_value_heads", cfg.num_key_value_heads);
  cfg.head_dim = regex_i64(text, "head_dim", cfg.head_dim);
  cfg.intermediate_size = regex_i64(text, "intermediate_size", cfg.intermediate_size);
  cfg.linear_key_head_dim = regex_i64(text, "linear_key_head_dim", cfg.linear_key_head_dim);
  cfg.linear_value_head_dim = regex_i64(text, "linear_value_head_dim", cfg.linear_value_head_dim);
  cfg.linear_num_key_heads = regex_i64(text, "linear_num_key_heads", cfg.linear_num_key_heads);
  cfg.linear_num_value_heads = regex_i64(text, "linear_num_value_heads", cfg.linear_num_value_heads);
  cfg.linear_conv_kernel_dim = regex_i64(text, "linear_conv_kernel_dim", cfg.linear_conv_kernel_dim);
  cfg.rms_norm_eps = regex_f64(text, "rms_norm_eps", cfg.rms_norm_eps);
  cfg.rope_theta = regex_f64(text, "rope_theta", cfg.rope_theta);
  cfg.partial_rotary_factor = regex_f64(text, "partial_rotary_factor", cfg.partial_rotary_factor);
  cfg.layer_types = parse_layer_types(text);
  if (cfg.layer_types.empty()) {
    cfg.layer_types.reserve(cfg.num_hidden_layers);
    for (int64_t i = 0; i < cfg.num_hidden_layers; ++i) {
      cfg.layer_types.push_back((i + 1) % 4 == 0 ? "full_attention" : "linear_attention");
    }
  }
  return cfg;
}

Qwen35TextModel::Qwen35TextModel(HfModelLoader loader) : loader_(std::move(loader)) {
  loader_.load_metadata();
  config_ = load_qwen35_text_config(loader_.model_dir());
}

void Qwen35TextModel::to(torch::Device device) {
  device_ = device;
  for (auto& item : weights_) {
    item.second = item.second.to(device_).contiguous();
  }
}

void Qwen35TextModel::set_weight_override(const std::string& name, torch::Tensor tensor) {
  weights_[name] = std::move(tensor);
}

std::vector<std::string> Qwen35TextModel::required_weight_names(int64_t max_layers) const {
  int64_t layers = max_layers < 0 ? config_.num_hidden_layers
                                  : std::min(max_layers, config_.num_hidden_layers);
  std::vector<std::string> names;
  names.reserve(static_cast<size_t>(8 + layers * 25));
  names.emplace_back("model.language_model.embed_tokens.weight");
  names.emplace_back("model.language_model.norm.weight");
  for (int64_t i = 0; i < layers; ++i) {
    std::string p = layer_prefix(i);
    names.emplace_back(p + "input_layernorm.weight");
    names.emplace_back(p + "post_attention_layernorm.weight");
    if (config_.layer_types.at(static_cast<size_t>(i)) == "full_attention") {
      std::string sa = p + "self_attn.";
      names.emplace_back(sa + "q_proj.weight");
      names.emplace_back(sa + "k_proj.weight");
      names.emplace_back(sa + "v_proj.weight");
      names.emplace_back(sa + "o_proj.weight");
      names.emplace_back(sa + "q_norm.weight");
      names.emplace_back(sa + "k_norm.weight");
    } else {
      std::string la = p + "linear_attn.";
      names.emplace_back(la + "in_proj_qkv.weight");
      names.emplace_back(la + "in_proj_z.weight");
      names.emplace_back(la + "in_proj_b.weight");
      names.emplace_back(la + "in_proj_a.weight");
      names.emplace_back(la + "conv1d.weight");
      names.emplace_back(la + "A_log");
      names.emplace_back(la + "dt_bias");
      names.emplace_back(la + "norm.weight");
      names.emplace_back(la + "out_proj.weight");
    }
    std::string mp = p + "mlp.";
    names.emplace_back(mp + "gate_proj.weight");
    names.emplace_back(mp + "up_proj.weight");
    names.emplace_back(mp + "down_proj.weight");
  }
  return names;
}

torch::Tensor Qwen35TextModel::weight(const std::string& name) {
  auto it = weights_.find(name);
  if (it != weights_.end()) {
    return it->second;
  }
  auto tensor = loader_.load_tensor(name).to(torch::kFloat32).to(device_).contiguous();
  auto inserted = weights_.emplace(name, tensor);
  return inserted.first->second;
}

torch::Tensor Qwen35TextModel::embed(const torch::Tensor& input_ids) {
  auto emb = weight("model.language_model.embed_tokens.weight");
  auto ids = input_ids.to(torch::TensorOptions().dtype(torch::kLong).device(emb.device()));
  return torch::nn::functional::embedding(ids, emb);
}

torch::Tensor Qwen35TextModel::rms_norm(const torch::Tensor& x, const torch::Tensor& w) const {
  auto xf = x.to(torch::kFloat32);
  auto out = xf * torch::rsqrt((xf * xf).mean(-1, true) + config_.rms_norm_eps);
  return (out * (1.0 + w.to(torch::kFloat32))).to(x.scalar_type());
}

torch::Tensor Qwen35TextModel::rms_norm_gated(const torch::Tensor& x,
                                             const torch::Tensor& gate,
                                             const torch::Tensor& w) const {
  auto xf = x.to(torch::kFloat32);
  auto out = xf * torch::rsqrt((xf * xf).mean(-1, true) + config_.rms_norm_eps);
  return ((out * w.to(torch::kFloat32)) * torch::silu(gate.to(torch::kFloat32))).to(x.scalar_type());
}

std::pair<torch::Tensor, torch::Tensor> Qwen35TextModel::rotary_embeddings(int64_t batch_size,
                                                                           int64_t seq_len,
                                                                           torch::Device device,
                                                                           torch::ScalarType dtype) const {
  int64_t rotary_dim = static_cast<int64_t>(config_.head_dim * config_.partial_rotary_factor);
  auto arange_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto inv_idx = torch::arange(0, rotary_dim, 2, arange_opts);
  auto inv_freq = torch::pow(config_.rope_theta, -(inv_idx / static_cast<double>(rotary_dim)));
  auto pos = torch::arange(0, seq_len, arange_opts);
  auto freqs = torch::ger(pos, inv_freq);
  auto emb = torch::cat({freqs, freqs}, -1).unsqueeze(0).expand({batch_size, seq_len, rotary_dim});
  return {torch::cos(emb).to(dtype), torch::sin(emb).to(dtype)};
}

torch::Tensor Qwen35TextModel::mlp(const torch::Tensor& x, int64_t layer_idx) {
  std::string p = layer_prefix(layer_idx) + "mlp.";
  auto gate = dense(x, weight(p + "gate_proj.weight"));
  auto up = dense(x, weight(p + "up_proj.weight"));
  return dense(torch::silu(gate) * up, weight(p + "down_proj.weight"));
}

torch::Tensor Qwen35TextModel::mlp_tensor_parallel(const torch::Tensor& x,
                                                   int64_t layer_idx,
                                                   const distributed::ParallelGroup& tensor_group) {
  std::string p = layer_prefix(layer_idx) + "mlp.";
  return distributed::tensor_parallel_mlp_swiglu(
      x,
      weight(p + "gate_proj.weight"),
      weight(p + "up_proj.weight"),
      weight(p + "down_proj.weight"),
      tensor_group);
}

torch::Tensor Qwen35TextModel::full_attention(const torch::Tensor& x, int64_t layer_idx) {
  std::string p = layer_prefix(layer_idx) + "self_attn.";
  int64_t b = x.size(0);
  int64_t s = x.size(1);
  int64_t h = config_.num_attention_heads;
  int64_t kvh = config_.num_key_value_heads;
  int64_t d = config_.head_dim;

  auto qg = dense(x, weight(p + "q_proj.weight")).view({b, s, h, d * 2});
  auto chunks = qg.chunk(2, -1);
  auto q = rms_norm(chunks[0], weight(p + "q_norm.weight")).transpose(1, 2);
  auto gate = chunks[1].reshape({b, s, h * d});
  auto k = rms_norm(dense(x, weight(p + "k_proj.weight")).view({b, s, kvh, d}), weight(p + "k_norm.weight")).transpose(1, 2);
  auto v = dense(x, weight(p + "v_proj.weight")).view({b, s, kvh, d}).transpose(1, 2);

  auto rope = rotary_embeddings(b, s, x.device(), x.scalar_type());
  auto cos = rope.first.unsqueeze(1);
  auto sin = rope.second.unsqueeze(1);
  int64_t rotary_dim = cos.size(-1);
  auto q_rot = q.index({Slice(), Slice(), Slice(), Slice(0, rotary_dim)});
  auto q_pass = q.index({Slice(), Slice(), Slice(), Slice(rotary_dim, None)});
  auto k_rot = k.index({Slice(), Slice(), Slice(), Slice(0, rotary_dim)});
  auto k_pass = k.index({Slice(), Slice(), Slice(), Slice(rotary_dim, None)});
  q = torch::cat(std::vector<torch::Tensor>{q_rot * cos + rotate_half(q_rot) * sin, q_pass}, -1);
  k = torch::cat(std::vector<torch::Tensor>{k_rot * cos + rotate_half(k_rot) * sin, k_pass}, -1);

  k = repeat_kv(k, h / kvh);
  v = repeat_kv(v, h / kvh);
  auto attn = torch::matmul(q, k.transpose(2, 3)) * (1.0 / std::sqrt(static_cast<double>(d)));
  auto mask = torch::ones({s, s}, torch::TensorOptions().dtype(torch::kBool).device(x.device())).triu(1);
  attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1.0e9);
  attn = torch::softmax(attn.to(torch::kFloat32), -1);
  attn = attn.to(v.scalar_type());
  auto out = torch::matmul(attn, v).transpose(1, 2).contiguous().reshape({b, s, h * d});
  out = out * torch::sigmoid(gate);
  return dense(out, weight(p + "o_proj.weight"));
}

torch::Tensor Qwen35TextModel::full_attention_tensor_parallel(const torch::Tensor& x,
                                                              int64_t layer_idx,
                                                              const distributed::ParallelGroup& tensor_group) {
  std::string p = layer_prefix(layer_idx) + "self_attn.";
  if (config_.num_attention_heads % tensor_group.world_size != 0) {
    throw std::invalid_argument("Qwen3.5 full attention TP requires num_attention_heads divisible by TP size");
  }
  if (tensor_group.world_size > config_.num_key_value_heads) {
    throw std::invalid_argument(
        "Qwen3.5 full attention TP size currently must be <= num_key_value_heads; use TP=2,DP=2 on 4xH20 for Qwen3.5-0.8B");
  }
  int64_t b = x.size(0);
  int64_t s = x.size(1);
  int64_t h = config_.num_attention_heads;
  int64_t h_local = h / tensor_group.world_size;
  int64_t kvh = config_.num_key_value_heads;
  int64_t d = config_.head_dim;

  auto q_weight = distributed::shard_dim(weight(p + "q_proj.weight"), 0, tensor_group.rank, tensor_group.world_size);
  auto qg = dense(x, q_weight).view({b, s, h_local, d * 2});
  auto chunks = qg.chunk(2, -1);
  auto q = rms_norm(chunks[0], weight(p + "q_norm.weight")).transpose(1, 2);
  auto gate = chunks[1].reshape({b, s, h_local * d});

  auto k = rms_norm(dense(x, weight(p + "k_proj.weight")).view({b, s, kvh, d}), weight(p + "k_norm.weight")).transpose(1, 2);
  auto v = dense(x, weight(p + "v_proj.weight")).view({b, s, kvh, d}).transpose(1, 2);

  auto rope = rotary_embeddings(b, s, x.device(), x.scalar_type());
  auto cos = rope.first.unsqueeze(1);
  auto sin = rope.second.unsqueeze(1);
  int64_t rotary_dim = cos.size(-1);
  auto q_rot = q.index({Slice(), Slice(), Slice(), Slice(0, rotary_dim)});
  auto q_pass = q.index({Slice(), Slice(), Slice(), Slice(rotary_dim, None)});
  auto k_rot = k.index({Slice(), Slice(), Slice(), Slice(0, rotary_dim)});
  auto k_pass = k.index({Slice(), Slice(), Slice(), Slice(rotary_dim, None)});
  q = torch::cat(std::vector<torch::Tensor>{q_rot * cos + rotate_half(q_rot) * sin, q_pass}, -1);
  k = torch::cat(std::vector<torch::Tensor>{k_rot * cos + rotate_half(k_rot) * sin, k_pass}, -1);

  k = repeat_kv(k, h / kvh).narrow(1, tensor_group.rank * h_local, h_local).contiguous();
  v = repeat_kv(v, h / kvh).narrow(1, tensor_group.rank * h_local, h_local).contiguous();
  auto attn = torch::matmul(q, k.transpose(2, 3)) * (1.0 / std::sqrt(static_cast<double>(d)));
  auto mask = torch::ones({s, s}, torch::TensorOptions().dtype(torch::kBool).device(x.device())).triu(1);
  attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1.0e9);
  attn = torch::softmax(attn.to(torch::kFloat32), -1);
  attn = attn.to(v.scalar_type());
  auto local = torch::matmul(attn, v).transpose(1, 2).contiguous().reshape({b, s, h_local * d});
  local = local * torch::sigmoid(gate);
  auto o_weight_shard = distributed::shard_dim(weight(p + "o_proj.weight"), 1, tensor_group.rank, tensor_group.world_size);
  auto partial = dense(local, o_weight_shard);
  if (tensor_group.world_size == 1) {
    return partial;
  }
  if (tensor_group.collectives == nullptr) {
    throw std::invalid_argument("Qwen3.5 full attention TP requires collectives");
  }
  return tensor_group.collectives->all_reduce(partial.contiguous(), distributed::ReduceOp::Sum, tensor_group.ranks);
}

torch::Tensor Qwen35TextModel::linear_attention(const torch::Tensor& x, int64_t layer_idx) {
  std::string p = layer_prefix(layer_idx) + "linear_attn.";
  int64_t bsz = x.size(0);
  int64_t seq = x.size(1);
  int64_t kh = config_.linear_num_key_heads;
  int64_t vh = config_.linear_num_value_heads;
  int64_t kd = config_.linear_key_head_dim;
  int64_t vd = config_.linear_value_head_dim;
  int64_t key_dim = kh * kd;
  int64_t value_dim = vh * vd;
  int64_t conv_dim = key_dim * 2 + value_dim;

  auto mixed = dense(x, weight(p + "in_proj_qkv.weight")).transpose(1, 2);
  auto conv_opts = torch::nn::functional::Conv1dFuncOptions().padding(config_.linear_conv_kernel_dim - 1).groups(conv_dim);
  mixed = torch::nn::functional::conv1d(mixed, weight(p + "conv1d.weight"), conv_opts);
  mixed = torch::silu(mixed.index({Slice(), Slice(), Slice(0, seq)})).transpose(1, 2);

  auto qkv = mixed.split({key_dim, key_dim, value_dim}, -1);
  auto query = qkv[0].reshape({bsz, seq, kh, kd});
  auto key = qkv[1].reshape({bsz, seq, kh, kd});
  auto value = qkv[2].reshape({bsz, seq, vh, vd});
  auto z = dense(x, weight(p + "in_proj_z.weight")).reshape({bsz, seq, vh, vd});
  auto beta = torch::sigmoid(dense(x, weight(p + "in_proj_b.weight")));
  auto a = dense(x, weight(p + "in_proj_a.weight"));
  auto g = -torch::exp(weight(p + "A_log")) * torch::softplus(a + weight(p + "dt_bias"));

  if (vh / kh > 1) {
    query = query.repeat_interleave(vh / kh, 2);
    key = key.repeat_interleave(vh / kh, 2);
  }

  query = l2norm(query).transpose(1, 2).contiguous().to(torch::kFloat32) * (1.0 / std::sqrt(static_cast<double>(kd)));
  key = l2norm(key).transpose(1, 2).contiguous().to(torch::kFloat32);
  value = value.transpose(1, 2).contiguous().to(torch::kFloat32);
  beta = beta.transpose(1, 2).contiguous().to(torch::kFloat32);
  g = g.transpose(1, 2).contiguous().to(torch::kFloat32);

  auto state = torch::zeros({bsz, vh, kd, vd}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
  std::vector<torch::Tensor> outs;
  outs.reserve(static_cast<size_t>(seq));
  for (int64_t t = 0; t < seq; ++t) {
    auto qt = query.index({Slice(), Slice(), t});
    auto kt = key.index({Slice(), Slice(), t});
    auto vt = value.index({Slice(), Slice(), t});
    auto gt = torch::exp(g.index({Slice(), Slice(), t})).unsqueeze(-1).unsqueeze(-1);
    auto bt = beta.index({Slice(), Slice(), t}).unsqueeze(-1);
    state = state * gt;
    auto kv_mem = (state * kt.unsqueeze(-1)).sum(-2);
    auto delta = (vt - kv_mem) * bt;
    state = state + kt.unsqueeze(-1) * delta.unsqueeze(-2);
    outs.push_back((state * qt.unsqueeze(-1)).sum(-2));
  }

  auto core = torch::stack(outs, 2).transpose(1, 2).contiguous().reshape({bsz * seq * vh, vd});
  auto zg = z.reshape({bsz * seq * vh, vd});
  core = rms_norm_gated(core, zg, weight(p + "norm.weight")).reshape({bsz, seq, value_dim});
  return dense(core, weight(p + "out_proj.weight"));
}

torch::Tensor Qwen35TextModel::linear_attention_tensor_parallel(const torch::Tensor& x,
                                                                int64_t layer_idx,
                                                                const distributed::ParallelGroup& tensor_group) {
  std::string p = layer_prefix(layer_idx) + "linear_attn.";
  if (config_.linear_num_key_heads % tensor_group.world_size != 0 ||
      config_.linear_num_value_heads % tensor_group.world_size != 0) {
    throw std::invalid_argument("Qwen3.5 linear attention TP requires linear heads divisible by TP size");
  }
  int64_t bsz = x.size(0);
  int64_t seq = x.size(1);
  int64_t kh = config_.linear_num_key_heads;
  int64_t vh = config_.linear_num_value_heads;
  int64_t kh_local = kh / tensor_group.world_size;
  int64_t vh_local = vh / tensor_group.world_size;
  int64_t kd = config_.linear_key_head_dim;
  int64_t vd = config_.linear_value_head_dim;
  int64_t key_dim = kh * kd;
  int64_t value_dim = vh * vd;
  int64_t key_local_dim = kh_local * kd;
  int64_t value_local_dim = vh_local * vd;
  int64_t conv_local_dim = key_local_dim * 2 + value_local_dim;
  int64_t key_begin = tensor_group.rank * key_local_dim;
  int64_t value_begin = tensor_group.rank * value_local_dim;

  auto q_weight = weight(p + "in_proj_qkv.weight").narrow(0, key_begin, key_local_dim);
  auto k_weight = weight(p + "in_proj_qkv.weight").narrow(0, key_dim + key_begin, key_local_dim);
  auto v_weight = weight(p + "in_proj_qkv.weight").narrow(0, key_dim * 2 + value_begin, value_local_dim);
  auto qkv_weight = torch::cat(std::vector<torch::Tensor>{q_weight, k_weight, v_weight}, 0).contiguous();
  auto mixed = dense(x, qkv_weight).transpose(1, 2);

  auto conv_q = weight(p + "conv1d.weight").narrow(0, key_begin, key_local_dim);
  auto conv_k = weight(p + "conv1d.weight").narrow(0, key_dim + key_begin, key_local_dim);
  auto conv_v = weight(p + "conv1d.weight").narrow(0, key_dim * 2 + value_begin, value_local_dim);
  auto conv_weight = torch::cat(std::vector<torch::Tensor>{conv_q, conv_k, conv_v}, 0).contiguous();
  auto conv_opts = torch::nn::functional::Conv1dFuncOptions().padding(config_.linear_conv_kernel_dim - 1).groups(conv_local_dim);
  mixed = torch::nn::functional::conv1d(mixed, conv_weight, conv_opts);
  mixed = torch::silu(mixed.index({Slice(), Slice(), Slice(0, seq)})).transpose(1, 2);

  auto qkv = mixed.split({key_local_dim, key_local_dim, value_local_dim}, -1);
  auto query = qkv[0].reshape({bsz, seq, kh_local, kd});
  auto key = qkv[1].reshape({bsz, seq, kh_local, kd});
  auto value = qkv[2].reshape({bsz, seq, vh_local, vd});
  auto z = dense(x, weight(p + "in_proj_z.weight").narrow(0, value_begin, value_local_dim)).reshape({bsz, seq, vh_local, vd});
  auto beta = torch::sigmoid(dense(x, weight(p + "in_proj_b.weight").narrow(0, tensor_group.rank * vh_local, vh_local)));
  auto a = dense(x, weight(p + "in_proj_a.weight").narrow(0, tensor_group.rank * vh_local, vh_local));
  auto a_log = weight(p + "A_log").narrow(0, tensor_group.rank * vh_local, vh_local);
  auto dt_bias = weight(p + "dt_bias").narrow(0, tensor_group.rank * vh_local, vh_local);
  auto g = -torch::exp(a_log) * torch::softplus(a + dt_bias);

  if (vh_local / kh_local > 1) {
    query = query.repeat_interleave(vh_local / kh_local, 2);
    key = key.repeat_interleave(vh_local / kh_local, 2);
  }

  query = l2norm(query).transpose(1, 2).contiguous().to(torch::kFloat32) * (1.0 / std::sqrt(static_cast<double>(kd)));
  key = l2norm(key).transpose(1, 2).contiguous().to(torch::kFloat32);
  value = value.transpose(1, 2).contiguous().to(torch::kFloat32);
  beta = beta.transpose(1, 2).contiguous().to(torch::kFloat32);
  g = g.transpose(1, 2).contiguous().to(torch::kFloat32);

  auto state = torch::zeros({bsz, vh_local, kd, vd}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
  std::vector<torch::Tensor> outs;
  outs.reserve(static_cast<size_t>(seq));
  for (int64_t t = 0; t < seq; ++t) {
    auto qt = query.index({Slice(), Slice(), t});
    auto kt = key.index({Slice(), Slice(), t});
    auto vt = value.index({Slice(), Slice(), t});
    auto gt = torch::exp(g.index({Slice(), Slice(), t})).unsqueeze(-1).unsqueeze(-1);
    auto bt = beta.index({Slice(), Slice(), t}).unsqueeze(-1);
    state = state * gt;
    auto kv_mem = (state * kt.unsqueeze(-1)).sum(-2);
    auto delta = (vt - kv_mem) * bt;
    state = state + kt.unsqueeze(-1) * delta.unsqueeze(-2);
    outs.push_back((state * qt.unsqueeze(-1)).sum(-2));
  }

  auto core = torch::stack(outs, 2).transpose(1, 2).contiguous().reshape({bsz * seq * vh_local, vd});
  auto zg = z.reshape({bsz * seq * vh_local, vd});
  core = rms_norm_gated(core, zg, weight(p + "norm.weight")).reshape({bsz, seq, value_local_dim});
  auto out_weight_shard = distributed::shard_dim(weight(p + "out_proj.weight"), 1, tensor_group.rank, tensor_group.world_size);
  auto partial = dense(core, out_weight_shard);
  if (tensor_group.world_size == 1) {
    return partial;
  }
  if (tensor_group.collectives == nullptr) {
    throw std::invalid_argument("Qwen3.5 linear attention TP requires collectives");
  }
  return tensor_group.collectives->all_reduce(partial.contiguous(), distributed::ReduceOp::Sum, tensor_group.ranks);
}

torch::Tensor Qwen35TextModel::forward_hidden(const torch::Tensor& input_ids, int64_t max_layers) {
  auto hidden = embed(input_ids);
  int64_t layers = max_layers < 0 ? config_.num_hidden_layers : std::min(max_layers, config_.num_hidden_layers);
  for (int64_t i = 0; i < layers; ++i) {
    std::string p = layer_prefix(i);
    auto residual = hidden;
    hidden = rms_norm(hidden, weight(p + "input_layernorm.weight"));
    if (config_.layer_types.at(static_cast<size_t>(i)) == "full_attention") {
      hidden = full_attention(hidden, i);
    } else {
      hidden = linear_attention(hidden, i);
    }
    hidden = residual + hidden;
    residual = hidden;
    hidden = rms_norm(hidden, weight(p + "post_attention_layernorm.weight"));
    hidden = residual + mlp(hidden, i);
  }
  return rms_norm(hidden, weight("model.language_model.norm.weight"));
}

torch::Tensor Qwen35TextModel::token_embeddings(const torch::Tensor& input_ids) {
  return embed(input_ids);
}

torch::Tensor Qwen35TextModel::forward_hidden_tensor_parallel(const torch::Tensor& input_ids,
                                                              const distributed::ParallelGroup& tensor_group,
                                                              int64_t max_layers) {
  auto hidden = embed(input_ids);
  int64_t layers = max_layers < 0 ? config_.num_hidden_layers : std::min(max_layers, config_.num_hidden_layers);
  return forward_hidden_range_tensor_parallel(hidden, 0, layers, tensor_group, true);
}

torch::Tensor Qwen35TextModel::forward_hidden_range_tensor_parallel(const torch::Tensor& hidden,
                                                                    int64_t layer_begin,
                                                                    int64_t layer_end,
                                                                    const distributed::ParallelGroup& tensor_group,
                                                                    bool apply_final_norm) {
  if (layer_begin < 0 || layer_end < layer_begin || layer_end > config_.num_hidden_layers) {
    throw std::invalid_argument("invalid Qwen3.5 layer range");
  }
  auto out = hidden;
  for (int64_t i = layer_begin; i < layer_end; ++i) {
    std::string p = layer_prefix(i);
    auto residual = out;
    out = rms_norm(out, weight(p + "input_layernorm.weight"));
    if (config_.layer_types.at(static_cast<size_t>(i)) == "full_attention") {
      out = full_attention_tensor_parallel(out, i, tensor_group);
    } else {
      out = linear_attention_tensor_parallel(out, i, tensor_group);
    }
    out = residual + out;
    residual = out;
    out = rms_norm(out, weight(p + "post_attention_layernorm.weight"));
    out = residual + mlp_tensor_parallel(out, i, tensor_group);
  }
  if (apply_final_norm) {
    out = rms_norm(out, weight("model.language_model.norm.weight"));
  }
  return out;
}

torch::Tensor Qwen35TextModel::forward_logits(const torch::Tensor& input_ids, int64_t max_layers) {
  auto hidden = forward_hidden(input_ids, max_layers);
  return dense(hidden, weight("model.language_model.embed_tokens.weight"));
}

}  // namespace cverl
