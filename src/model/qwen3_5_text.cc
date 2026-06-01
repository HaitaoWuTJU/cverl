#include "cverl/model/qwen3_5_text.h"
#include "cverl/distributed/expert_parallel.h"
#include "cverl/model/qwen_linear_attention_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <tuple>

#include <torch/nn/functional/conv.h>
#include <torch/nn/functional/embedding.h>
#include <torch/csrc/autograd/custom_function.h>

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

std::string regex_string(const std::string& text, const std::string& key, const std::string& fallback = {}) {
  std::smatch match;
  std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
  if (std::regex_search(text, match, pattern)) {
    return match[1].str();
  }
  return fallback;
}

bool regex_bool(const std::string& text, const std::string& key, bool fallback) {
  std::smatch match;
  std::regex pattern("\"" + key + "\"\\s*:\\s*(true|false)");
  if (std::regex_search(text, match, pattern)) {
    return match[1].str() == "true";
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

std::string tensor_cache_key(int64_t seq_len, torch::Device device, torch::ScalarType dtype) {
  return std::to_string(seq_len) + "|" + device.str() + "|" + c10::toString(dtype);
}

int64_t normalize_dim_for_tensor(const torch::Tensor& tensor, int64_t dim) {
  if (dim < 0) {
    dim += tensor.dim();
  }
  if (dim < 0 || dim >= tensor.dim()) {
    throw std::invalid_argument("tensor dimension out of range");
  }
  return dim;
}

int64_t group_local_rank(const distributed::ParallelGroup& group, const char* name) {
  if (group.world_size <= 0) {
    throw std::invalid_argument(std::string(name) + " requires positive world_size");
  }
  if (static_cast<int64_t>(group.ranks.size()) != group.world_size) {
    throw std::invalid_argument(std::string(name) + " ranks must match world_size");
  }
  for (int64_t i = 0; i < group.world_size; ++i) {
    if (group.ranks.at(static_cast<size_t>(i)) == group.rank) {
      return i;
    }
  }
  if (group.rank >= 0 && group.rank < group.world_size) {
    return group.rank;
  }
  throw std::invalid_argument(std::string(name) + " rank must be a group-local rank or a member of ranks");
}

torch::Tensor ring_exchange_rank_order(const torch::Tensor& local,
                                       const distributed::ParallelGroup& context_group,
                                       int64_t sequence_dim) {
  const int64_t dim = normalize_dim_for_tensor(local, sequence_dim);
  const int64_t context_rank = group_local_rank(context_group, "Qwen3.5 CP ring exchange");
  auto ring = distributed::context_parallel_ring_exchange_autograd(
      local.contiguous(), *context_group.collectives, context_group.ranks, context_rank, dim);
  if (context_group.world_size <= 1) {
    return ring;
  }
  const int64_t shard = local.size(dim);
  const auto schedule = distributed::context_parallel_ring_schedule(
      context_group.ranks, context_group.ranks.at(static_cast<size_t>(context_rank)));
  std::vector<torch::Tensor> rank_order(static_cast<size_t>(context_group.world_size));
  for (int64_t ring_index = 0; ring_index < static_cast<int64_t>(schedule.size()); ++ring_index) {
    const int64_t rank_index =
        distributed::context_parallel_group_index(context_group.ranks, schedule[static_cast<size_t>(ring_index)].kv_rank);
    rank_order[static_cast<size_t>(rank_index)] = ring.narrow(dim, ring_index * shard, shard);
  }
  return torch::cat(rank_order, dim).contiguous();
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

torch::Tensor local_or_shard_dim(const torch::Tensor& tensor,
                                 int64_t dim,
                                 int64_t full_size,
                                 int64_t rank,
                                 int64_t world_size,
                                 const std::string& name) {
  if (world_size == 1) {
    return tensor;
  }
  if (dim < 0) {
    dim += tensor.dim();
  }
  if (dim < 0 || dim >= tensor.dim()) {
    throw std::invalid_argument("invalid TP shard dim for " + name);
  }
  if (full_size % world_size != 0) {
    throw std::invalid_argument("full tensor dimension is not divisible by TP size for " + name);
  }
  const int64_t local_size = full_size / world_size;
  if (tensor.size(dim) == local_size) {
    return tensor;
  }
  if (tensor.size(dim) == full_size) {
    return tensor.narrow(dim, rank * local_size, local_size).contiguous();
  }
  throw std::invalid_argument("unexpected TP tensor shape for " + name);
}

torch::Tensor all_reduce_sum_preserve_local_grad(const torch::Tensor& local,
                                                 const distributed::ParallelGroup& tensor_group) {
  if (tensor_group.world_size == 1) {
    return local;
  }
  if (tensor_group.collectives == nullptr) {
    throw std::invalid_argument("TP all-reduce requires collectives");
  }
  auto reduced = tensor_group.collectives->all_reduce(local.contiguous(), distributed::ReduceOp::Sum, tensor_group.ranks);
  return local + (reduced - local).detach();
}

int64_t linear_attention_chunk_size() {
  const char* env = std::getenv("CVERL_LINEAR_ATTN_CHUNK_SIZE");
  if (env == nullptr || *env == '\0') {
    return 16;
  }
  int64_t value = std::stoll(env);
  if (value <= 0) {
    throw std::invalid_argument("CVERL_LINEAR_ATTN_CHUNK_SIZE must be positive");
  }
  return value;
}

std::tuple<torch::Tensor, torch::Tensor> linear_attention_chunked_recurrent_with_state(const torch::Tensor& query,
                                                                                       const torch::Tensor& key,
                                                                                       const torch::Tensor& value,
                                                                                       const torch::Tensor& beta,
                                                                                       const torch::Tensor& g,
                                                                                       const torch::Tensor& initial_state,
                                                                                       int64_t kd,
                                                                                       int64_t vd) {
  const int64_t bsz = query.size(0);
  const int64_t heads = query.size(1);
  const int64_t seq = query.size(2);
  auto state = initial_state.defined()
                   ? initial_state.to(torch::kFloat32).contiguous()
                   : torch::zeros({bsz, heads, kd, vd},
                                  torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));
  if (state.sizes() != torch::IntArrayRef({bsz, heads, kd, vd})) {
    throw std::invalid_argument("linear attention initial state shape mismatch");
  }
  if (seq == 0) {
    return {torch::empty({bsz, heads, 0, vd}, query.options().dtype(torch::kFloat32)), state};
  }
  std::vector<torch::Tensor> chunk_outputs;
  const int64_t chunk_size = linear_attention_chunk_size();
  chunk_outputs.reserve(static_cast<size_t>((seq + chunk_size - 1) / chunk_size));
  for (int64_t begin = 0; begin < seq; begin += chunk_size) {
    const int64_t end = std::min(seq, begin + chunk_size);
    std::vector<torch::Tensor> outs;
    outs.reserve(static_cast<size_t>(end - begin));
    for (int64_t t = begin; t < end; ++t) {
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
    chunk_outputs.push_back(torch::stack(outs, 2));
  }
  return {torch::cat(chunk_outputs, 2), state};
}

torch::Tensor linear_attention_chunked_recurrent(const torch::Tensor& query,
                                                 const torch::Tensor& key,
                                                 const torch::Tensor& value,
                                                 const torch::Tensor& beta,
                                                 const torch::Tensor& g,
                                                 int64_t kd,
                                                 int64_t vd) {
  return std::get<0>(
      linear_attention_chunked_recurrent_with_state(query, key, value, beta, g, torch::Tensor(), kd, vd));
}

bool use_cuda_linear_attention_kernel(const torch::Tensor& query, int64_t kd, int64_t vd) {
  const char* env = std::getenv("CVERL_LINEAR_ATTN_BACKEND");
  const std::string backend = env == nullptr ? "cuda" : std::string(env);
  if (backend == "eager" || backend == "chunked") {
    return false;
  }
  if (backend != "cuda" && backend != "auto") {
    throw std::invalid_argument("CVERL_LINEAR_ATTN_BACKEND must be cuda|auto|chunked|eager");
  }
  return qwen_linear_attention_cuda_available() && query.is_cuda() && query.scalar_type() == torch::kFloat32 &&
         kd == 128 && vd == 128;
}

enum class CudaLinearAttentionStateMode {
  None = 0,
  Full = 1,
  Chunk = 2,
};

CudaLinearAttentionStateMode cuda_linear_attention_state_mode() {
  const char* mode_env = std::getenv("CVERL_LINEAR_ATTN_STATE_MODE");
  if (mode_env != nullptr && *mode_env != '\0') {
    const std::string mode(mode_env);
    if (mode == "none" || mode == "recompute") {
      return CudaLinearAttentionStateMode::None;
    }
    if (mode == "full") {
      return CudaLinearAttentionStateMode::Full;
    }
    if (mode == "chunk" || mode == "checkpoint") {
      return CudaLinearAttentionStateMode::Chunk;
    }
    throw std::invalid_argument("CVERL_LINEAR_ATTN_STATE_MODE must be none|recompute|full|chunk|checkpoint");
  }

  const char* save_env = std::getenv("CVERL_LINEAR_ATTN_SAVE_STATES");
  if (save_env != nullptr && std::string(save_env) == "1") {
    return CudaLinearAttentionStateMode::Full;
  }
  return CudaLinearAttentionStateMode::Chunk;
}

int64_t cuda_linear_attention_checkpoint_interval(const torch::Tensor& query, int64_t kd, int64_t vd) {
  const char* env = std::getenv("CVERL_LINEAR_ATTN_CHECKPOINT_INTERVAL");
  int64_t value = (env == nullptr || *env == '\0') ? linear_attention_chunk_size() : std::stoll(env);
  if (value <= 0) {
    throw std::invalid_argument("CVERL_LINEAR_ATTN_CHECKPOINT_INTERVAL must be positive");
  }
  const char* budget_env = std::getenv("CVERL_LINEAR_ATTN_CHECKPOINT_MAX_BYTES_MB");
  if (budget_env != nullptr && *budget_env != '\0') {
    const int64_t budget_mb = std::stoll(budget_env);
    if (budget_mb <= 0) {
      throw std::invalid_argument("CVERL_LINEAR_ATTN_CHECKPOINT_MAX_BYTES_MB must be positive");
    }
    const int64_t bytes_budget = budget_mb * 1024LL * 1024LL;
    const int64_t bytes_per_checkpoint = query.size(0) * query.size(1) * kd * vd * query.element_size();
    if (bytes_per_checkpoint > 0) {
      const int64_t max_checkpoints = std::max<int64_t>(1, bytes_budget / bytes_per_checkpoint);
      const int64_t seq = query.size(2);
      const int64_t budget_interval = std::max<int64_t>(1, (seq + max_checkpoints - 1) / max_checkpoints);
      value = std::max(value, budget_interval);
    }
  }
  return value;
}

class QwenLinearAttentionCudaFunction
    : public torch::autograd::Function<QwenLinearAttentionCudaFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::Tensor query,
                               torch::Tensor key,
                               torch::Tensor value,
                               torch::Tensor beta,
                               torch::Tensor g) {
    const auto state_mode = cuda_linear_attention_state_mode();
    const int64_t checkpoint_interval =
        state_mode == CudaLinearAttentionStateMode::Chunk
            ? cuda_linear_attention_checkpoint_interval(query, query.size(3), value.size(3))
            : 0;
    std::tuple<torch::Tensor, torch::Tensor> result;
    if (state_mode == CudaLinearAttentionStateMode::Chunk) {
      result = qwen_linear_attention_cuda_forward_checkpointed(
          query.contiguous(),
          key.contiguous(),
          value.contiguous(),
          beta.contiguous(),
          g.contiguous(),
          checkpoint_interval);
    } else {
      result = qwen_linear_attention_cuda_forward(
          query.contiguous(),
          key.contiguous(),
          value.contiguous(),
          beta.contiguous(),
          g.contiguous(),
          state_mode == CudaLinearAttentionStateMode::Full);
    }
    auto out = std::get<0>(result);
    auto states = std::get<1>(result);
    ctx->saved_data["state_mode"] = static_cast<int64_t>(state_mode);
    ctx->saved_data["checkpoint_interval"] = checkpoint_interval;
    ctx->save_for_backward({query, key, value, beta, g, states});
    return out;
  }

  static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx,
                                               torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    const auto state_mode =
        static_cast<CudaLinearAttentionStateMode>(ctx->saved_data["state_mode"].toInt());
    const int64_t checkpoint_interval = ctx->saved_data["checkpoint_interval"].toInt();
    std::vector<torch::Tensor> grads;
    if (state_mode == CudaLinearAttentionStateMode::Full) {
      grads = qwen_linear_attention_cuda_backward(
          grad_outputs[0].contiguous(),
          saved[0].contiguous(),
          saved[1].contiguous(),
          saved[2].contiguous(),
          saved[3].contiguous(),
          saved[4].contiguous(),
          saved[5].contiguous());
    } else if (state_mode == CudaLinearAttentionStateMode::Chunk) {
      grads = qwen_linear_attention_cuda_backward_checkpointed(
          grad_outputs[0].contiguous(),
          saved[0].contiguous(),
          saved[1].contiguous(),
          saved[2].contiguous(),
          saved[3].contiguous(),
          saved[4].contiguous(),
          saved[5].contiguous(),
          checkpoint_interval);
    } else {
      grads = qwen_linear_attention_cuda_backward_recompute(
          grad_outputs[0].contiguous(),
          saved[0].contiguous(),
          saved[1].contiguous(),
          saved[2].contiguous(),
          saved[3].contiguous(),
          saved[4].contiguous());
    }
    return {grads[0], grads[1], grads[2], grads[3], grads[4]};
  }
};

torch::Tensor linear_attention_recurrent(const torch::Tensor& query,
                                         const torch::Tensor& key,
                                         const torch::Tensor& value,
                                         const torch::Tensor& beta,
                                         const torch::Tensor& g,
                                         int64_t kd,
                                         int64_t vd) {
  if (use_cuda_linear_attention_kernel(query, kd, vd)) {
    return QwenLinearAttentionCudaFunction::apply(query, key, value, beta, g);
  }
  return linear_attention_chunked_recurrent(query, key, value, beta, g, kd, vd);
}

std::tuple<torch::Tensor, torch::Tensor> linear_attention_recurrent_with_state(const torch::Tensor& query,
                                                                               const torch::Tensor& key,
                                                                               const torch::Tensor& value,
                                                                               const torch::Tensor& beta,
                                                                               const torch::Tensor& g,
                                                                               const torch::Tensor& initial_state,
                                                                               int64_t kd,
                                                                               int64_t vd) {
  return linear_attention_chunked_recurrent_with_state(query, key, value, beta, g, initial_state, kd, vd);
}

class LinearAttentionInitialStateRecvFunction final
    : public torch::autograd::Function<LinearAttentionInitialStateRecvFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::Tensor state_like,
                               int64_t collectives_addr,
                               int64_t prev_peer) {
    auto* collectives =
        reinterpret_cast<distributed::Collectives*>(static_cast<uintptr_t>(collectives_addr));
    if (collectives == nullptr) {
      throw std::invalid_argument("linear attention CP initial state recv requires collectives");
    }
    ctx->saved_data["collectives_addr"] = collectives_addr;
    ctx->saved_data["prev_peer"] = prev_peer;
    return collectives->recv_like(state_like.contiguous(), prev_peer).contiguous();
  }

  static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx,
                                               torch::autograd::tensor_list grad_outputs) {
    auto* collectives = reinterpret_cast<distributed::Collectives*>(
        static_cast<uintptr_t>(ctx->saved_data["collectives_addr"].toInt()));
    const int64_t prev_peer = ctx->saved_data["prev_peer"].toInt();
    if (grad_outputs.at(0).defined()) {
      collectives->send(grad_outputs.at(0).contiguous(), prev_peer);
    }
    return {torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }
};

class LinearAttentionFinalStateSendFunction final
    : public torch::autograd::Function<LinearAttentionFinalStateSendFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::Tensor local_output,
                               torch::Tensor final_state,
                               int64_t collectives_addr,
                               int64_t next_peer) {
    auto* collectives =
        reinterpret_cast<distributed::Collectives*>(static_cast<uintptr_t>(collectives_addr));
    if (collectives == nullptr) {
      throw std::invalid_argument("linear attention CP final state send requires collectives");
    }
    ctx->saved_data["collectives_addr"] = collectives_addr;
    ctx->saved_data["next_peer"] = next_peer;
    ctx->save_for_backward({final_state});
    collectives->send(final_state.contiguous(), next_peer);
    return local_output;
  }

  static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx,
                                               torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto* collectives = reinterpret_cast<distributed::Collectives*>(
        static_cast<uintptr_t>(ctx->saved_data["collectives_addr"].toInt()));
    const int64_t next_peer = ctx->saved_data["next_peer"].toInt();
    auto grad_final = collectives->recv_like(saved.at(0).contiguous(), next_peer).contiguous();
    return {grad_outputs.at(0), grad_final, torch::Tensor(), torch::Tensor()};
  }
};

torch::Tensor linear_attention_context_initial_state(const torch::Tensor& state_like,
                                                     const distributed::ParallelGroup& context_group,
                                                     int64_t context_rank) {
  if (context_rank == 0) {
    return torch::Tensor();
  }
  const int64_t prev_peer = context_group.ranks.at(static_cast<size_t>(context_rank - 1));
  const auto collectives_addr = static_cast<int64_t>(reinterpret_cast<uintptr_t>(context_group.collectives));
  return LinearAttentionInitialStateRecvFunction::apply(
      state_like.contiguous().set_requires_grad(true), collectives_addr, prev_peer);
}

torch::Tensor linear_attention_attach_final_state_carry(const torch::Tensor& local_output,
                                                        const torch::Tensor& final_state,
                                                        const distributed::ParallelGroup& context_group,
                                                        int64_t context_rank) {
  if (context_rank + 1 >= context_group.world_size) {
    return local_output;
  }
  const int64_t next_peer = context_group.ranks.at(static_cast<size_t>(context_rank + 1));
  const auto collectives_addr = static_cast<int64_t>(reinterpret_cast<uintptr_t>(context_group.collectives));
  return LinearAttentionFinalStateSendFunction::apply(local_output, final_state, collectives_addr, next_peer);
}

std::string layer_prefix(const Qwen35TextConfig& cfg, int64_t layer_idx) {
  const char* root = cfg.is_qwen3_moe() ? "model.layers." : "model.language_model.layers.";
  return std::string(root) + std::to_string(layer_idx) + ".";
}

std::string embed_weight_name(const Qwen35TextConfig& cfg) {
  return cfg.is_qwen3_moe() ? "model.embed_tokens.weight" : "model.language_model.embed_tokens.weight";
}

std::string final_norm_weight_name(const Qwen35TextConfig& cfg) {
  return cfg.is_qwen3_moe() ? "model.norm.weight" : "model.language_model.norm.weight";
}

std::string lm_head_weight_name(const Qwen35TextConfig& cfg) {
  return (cfg.is_qwen3_moe() && !cfg.tie_word_embeddings) ? "lm_head.weight" : embed_weight_name(cfg);
}

}  // namespace

Qwen35TextConfig load_qwen35_text_config(const std::string& model_dir) {
  std::string text = read_file(std::filesystem::path(model_dir) / "config.json");
  Qwen35TextConfig cfg;
  cfg.model_type = regex_string(text, "model_type", "qwen3_5");
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
  cfg.moe_intermediate_size = regex_i64(text, "moe_intermediate_size", cfg.moe_intermediate_size);
  cfg.num_experts = regex_i64(text, "num_experts", cfg.num_experts);
  cfg.num_experts_per_tok = regex_i64(text, "num_experts_per_tok", cfg.num_experts_per_tok);
  cfg.rms_norm_eps = regex_f64(text, "rms_norm_eps", cfg.rms_norm_eps);
  cfg.rope_theta = regex_f64(text, "rope_theta", cfg.rope_theta);
  cfg.partial_rotary_factor = regex_f64(text, "partial_rotary_factor", cfg.partial_rotary_factor);
  cfg.norm_topk_prob = regex_bool(text, "norm_topk_prob", cfg.norm_topk_prob);
  cfg.tie_word_embeddings = regex_bool(text, "tie_word_embeddings", cfg.tie_word_embeddings);
  cfg.layer_types = parse_layer_types(text);
  if (cfg.is_qwen3_moe()) {
    cfg.partial_rotary_factor = 1.0;
    cfg.layer_types.assign(static_cast<size_t>(cfg.num_hidden_layers), "full_attention");
    if (cfg.moe_intermediate_size <= 0 || cfg.num_experts <= 0 || cfg.num_experts_per_tok <= 0) {
      throw std::runtime_error("qwen3_moe config requires moe_intermediate_size/num_experts/num_experts_per_tok");
    }
    return cfg;
  }
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
  causal_mask_cache_.clear();
  rotary_cache_.clear();
}

void Qwen35TextModel::set_weight_override(const std::string& name, torch::Tensor tensor) {
  weights_[name] = std::move(tensor);
}

std::vector<std::string> Qwen35TextModel::required_weight_names(int64_t max_layers) const {
  int64_t layers = max_layers < 0 ? config_.num_hidden_layers
                                  : std::min(max_layers, config_.num_hidden_layers);
  std::vector<std::string> names;
  names.reserve(static_cast<size_t>(8 + layers * (config_.is_qwen3_moe() ? (12 + config_.num_experts * 3) : 25)));
  names.emplace_back(embed_weight_name(config_));
  names.emplace_back(final_norm_weight_name(config_));
  if (config_.is_qwen3_moe() && !config_.tie_word_embeddings) {
    names.emplace_back("lm_head.weight");
  }
  for (int64_t i = 0; i < layers; ++i) {
    std::string p = layer_prefix(config_, i);
    names.emplace_back(p + "input_layernorm.weight");
    names.emplace_back(p + "post_attention_layernorm.weight");
    if (config_.is_qwen3_moe()) {
      std::string sa = p + "self_attn.";
      names.emplace_back(sa + "q_proj.weight");
      names.emplace_back(sa + "k_proj.weight");
      names.emplace_back(sa + "v_proj.weight");
      names.emplace_back(sa + "o_proj.weight");
      names.emplace_back(sa + "q_norm.weight");
      names.emplace_back(sa + "k_norm.weight");
      names.emplace_back(p + "mlp.gate.weight");
      for (int64_t expert = 0; expert < config_.num_experts; ++expert) {
        std::string ep = p + "mlp.experts." + std::to_string(expert) + ".";
        names.emplace_back(ep + "gate_proj.weight");
        names.emplace_back(ep + "up_proj.weight");
        names.emplace_back(ep + "down_proj.weight");
      }
    } else if (config_.layer_types.at(static_cast<size_t>(i)) == "full_attention") {
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
    if (!config_.is_qwen3_moe()) {
      std::string mp = p + "mlp.";
      names.emplace_back(mp + "gate_proj.weight");
      names.emplace_back(mp + "up_proj.weight");
      names.emplace_back(mp + "down_proj.weight");
    }
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
  auto emb = weight(embed_weight_name(config_));
  auto ids = input_ids.to(torch::TensorOptions().dtype(torch::kLong).device(emb.device()));
  return torch::nn::functional::embedding(ids, emb);
}

torch::Tensor Qwen35TextModel::rms_norm(const torch::Tensor& x, const torch::Tensor& w) const {
  auto xf = x.to(torch::kFloat32);
  auto out = xf * torch::rsqrt((xf * xf).mean(-1, true) + config_.rms_norm_eps);
  auto scale = config_.is_qwen3_moe() ? w.to(torch::kFloat32) : (1.0 + w.to(torch::kFloat32));
  return (out * scale).to(x.scalar_type());
}

torch::Tensor Qwen35TextModel::rms_norm_gated(const torch::Tensor& x,
                                              const torch::Tensor& gate,
                                              const torch::Tensor& w) const {
  auto xf = x.to(torch::kFloat32);
  auto out = xf * torch::rsqrt((xf * xf).mean(-1, true) + config_.rms_norm_eps);
  return ((out * w.to(torch::kFloat32)) * torch::silu(gate.to(torch::kFloat32))).to(x.scalar_type());
}

torch::Tensor Qwen35TextModel::causal_mask(int64_t seq_len, torch::Device device) const {
  const std::string key = std::to_string(seq_len) + "|" + device.str();
  auto it = causal_mask_cache_.find(key);
  if (it != causal_mask_cache_.end()) {
    return it->second;
  }
  auto mask = torch::ones({seq_len, seq_len}, torch::TensorOptions().dtype(torch::kBool).device(device)).triu(1);
  auto inserted = causal_mask_cache_.emplace(key, mask);
  return inserted.first->second;
}

std::pair<torch::Tensor, torch::Tensor> Qwen35TextModel::rotary_embeddings(int64_t batch_size,
                                                                           int64_t seq_len,
                                                                           torch::Device device,
                                                                           torch::ScalarType dtype) const {
  return rotary_embeddings_range(batch_size, 0, seq_len, device, dtype);
}

std::pair<torch::Tensor, torch::Tensor> Qwen35TextModel::rotary_embeddings_range(int64_t batch_size,
                                                                                 int64_t position_begin,
                                                                                 int64_t seq_len,
                                                                                 torch::Device device,
                                                                                 torch::ScalarType dtype) const {
  if (position_begin < 0 || seq_len < 0) {
    throw std::invalid_argument("rotary embedding range must be non-negative");
  }
  int64_t rotary_dim = static_cast<int64_t>(config_.head_dim * config_.partial_rotary_factor);
  const std::string key = tensor_cache_key(position_begin + seq_len, device, dtype) + "|" + std::to_string(rotary_dim);
  auto it = rotary_cache_.find(key);
  if (it != rotary_cache_.end()) {
    return {it->second.first.narrow(1, position_begin, seq_len).expand({batch_size, seq_len, rotary_dim}),
            it->second.second.narrow(1, position_begin, seq_len).expand({batch_size, seq_len, rotary_dim})};
  }
  auto arange_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto inv_idx = torch::arange(0, rotary_dim, 2, arange_opts);
  auto inv_freq = torch::pow(config_.rope_theta, -(inv_idx / static_cast<double>(rotary_dim)));
  auto pos = torch::arange(0, position_begin + seq_len, arange_opts);
  auto freqs = torch::ger(pos, inv_freq);
  auto emb = torch::cat({freqs, freqs}, -1).unsqueeze(0);
  auto cos = torch::cos(emb).to(dtype).contiguous();
  auto sin = torch::sin(emb).to(dtype).contiguous();
  auto inserted = rotary_cache_.emplace(key, std::make_pair(cos, sin));
  return {inserted.first->second.first.narrow(1, position_begin, seq_len).expand({batch_size, seq_len, rotary_dim}),
          inserted.first->second.second.narrow(1, position_begin, seq_len).expand({batch_size, seq_len, rotary_dim})};
}

torch::Tensor Qwen35TextModel::mlp(const torch::Tensor& x, int64_t layer_idx) {
  std::string p = layer_prefix(config_, layer_idx) + "mlp.";
  if (config_.is_qwen3_moe()) {
    auto flat = x.reshape({-1, config_.hidden_size});
    auto router_logits = dense(flat, weight(p + "gate.weight")).to(torch::kFloat32);
    auto routing = torch::softmax(router_logits, -1);
    auto topk = routing.topk(config_.num_experts_per_tok, -1, true, false);
    auto topk_weights = std::get<0>(topk);
    auto topk_indices = std::get<1>(topk);
    if (config_.norm_topk_prob) {
      topk_weights = topk_weights / topk_weights.sum(-1, true).clamp_min(1.0e-12);
    }
    auto out = torch::zeros_like(flat);
    for (int64_t expert = 0; expert < config_.num_experts; ++expert) {
      auto positions = torch::nonzero(topk_indices == expert);
      if (positions.numel() == 0) {
        continue;
      }
      auto token_indices = positions.index({Slice(), 0}).to(torch::TensorOptions().dtype(torch::kLong).device(flat.device()));
      auto slot_indices = positions.index({Slice(), 1}).to(torch::TensorOptions().dtype(torch::kLong).device(flat.device()));
      auto selected = flat.index_select(0, token_indices);
      std::string ep = p + "experts." + std::to_string(expert) + ".";
      auto gate = dense(selected, weight(ep + "gate_proj.weight"));
      auto up = dense(selected, weight(ep + "up_proj.weight"));
      auto expert_out = dense(torch::silu(gate) * up, weight(ep + "down_proj.weight"));
      auto expert_weight = topk_weights.index({token_indices, slot_indices}).unsqueeze(-1).to(expert_out.scalar_type());
      out.index_add_(0, token_indices, expert_out * expert_weight);
    }
    return out.reshape_as(x);
  }
  auto gate = dense(x, weight(p + "gate_proj.weight"));
  auto up = dense(x, weight(p + "up_proj.weight"));
  return dense(torch::silu(gate) * up, weight(p + "down_proj.weight"));
}

torch::Tensor Qwen35TextModel::mlp_expert_parallel(const torch::Tensor& x,
                                                   int64_t layer_idx,
                                                   const distributed::ParallelGroup& tensor_group,
                                                   const distributed::ParallelGroup& expert_group) {
  if (!config_.is_qwen3_moe()) {
    return mlp(x, layer_idx);
  }
  if (tensor_group.world_size <= 0 || tensor_group.rank < 0 || tensor_group.rank >= tensor_group.world_size) {
    throw std::invalid_argument("Qwen3 MoE TP+EP MLP has invalid TP rank/world_size");
  }

  std::string p = layer_prefix(config_, layer_idx) + "mlp.";
  auto flat = x.reshape({-1, config_.hidden_size});
  auto router_logits = dense(flat, weight(p + "gate.weight")).to(torch::kFloat32);
  auto routing = torch::softmax(router_logits, -1);
  auto topk = routing.topk(config_.num_experts_per_tok, -1, true, false);
  auto topk_weights = std::get<0>(topk);
  auto topk_indices = std::get<1>(topk);
  if (config_.norm_topk_prob) {
    topk_weights = topk_weights / topk_weights.sum(-1, true).clamp_min(1.0e-12);
  }
  if (config_.moe_intermediate_size % tensor_group.world_size != 0) {
    throw std::invalid_argument("Qwen3 MoE expert intermediate size must be divisible by TP size");
  }
  const int64_t intermediate_local = config_.moe_intermediate_size / tensor_group.world_size;

  if (expert_group.world_size == 1) {
    auto out = torch::zeros_like(flat);
    for (int64_t expert = 0; expert < config_.num_experts; ++expert) {
      auto positions = torch::nonzero(topk_indices == expert);
      if (positions.numel() == 0) {
        continue;
      }
      auto token_indices = positions.index({Slice(), 0}).to(torch::TensorOptions().dtype(torch::kLong).device(flat.device()));
      auto slot_indices = positions.index({Slice(), 1}).to(torch::TensorOptions().dtype(torch::kLong).device(flat.device()));
      auto selected = flat.index_select(0, token_indices);
      std::string ep = p + "experts." + std::to_string(expert) + ".";
      auto gate_weight = local_or_shard_dim(weight(ep + "gate_proj.weight"),
                                            0,
                                            config_.moe_intermediate_size,
                                            tensor_group.rank,
                                            tensor_group.world_size,
                                            ep + "gate_proj.weight");
      auto up_weight = local_or_shard_dim(weight(ep + "up_proj.weight"),
                                          0,
                                          config_.moe_intermediate_size,
                                          tensor_group.rank,
                                          tensor_group.world_size,
                                          ep + "up_proj.weight");
      auto down_weight = local_or_shard_dim(weight(ep + "down_proj.weight"),
                                            1,
                                            config_.moe_intermediate_size,
                                            tensor_group.rank,
                                            tensor_group.world_size,
                                            ep + "down_proj.weight");
      if (gate_weight.size(0) != intermediate_local || up_weight.size(0) != intermediate_local ||
          down_weight.size(1) != intermediate_local) {
        throw std::invalid_argument("Qwen3 MoE TP expert shard shape mismatch");
      }
      auto gate = dense(selected, gate_weight);
      auto up = dense(selected, up_weight);
      auto expert_out = dense(torch::silu(gate) * up, down_weight);
      auto expert_weight = topk_weights.index({token_indices, slot_indices}).unsqueeze(-1).to(expert_out.scalar_type());
      out.index_add_(0, token_indices, expert_out * expert_weight);
    }
    if (tensor_group.world_size > 1) {
      out = all_reduce_sum_preserve_local_grad(out.contiguous(), tensor_group);
    }
    return out.reshape_as(x);
  }

  if (expert_group.collectives == nullptr) {
    throw std::invalid_argument("Qwen3 MoE EP MLP requires collectives");
  }
  if (config_.num_experts % expert_group.world_size != 0) {
    throw std::invalid_argument("Qwen3 MoE expert count must be divisible by EP size");
  }
  if (expert_group.rank < 0 || expert_group.rank >= expert_group.world_size) {
    throw std::invalid_argument("invalid Qwen3 MoE EP rank");
  }

  const int64_t experts_per_rank = config_.num_experts / expert_group.world_size;
  const int64_t expert_begin = expert_group.rank * experts_per_rank;
  const int64_t expert_end = expert_begin + experts_per_rank;
  const int64_t route_count = flat.size(0) * config_.num_experts_per_tok;
  auto long_options = torch::TensorOptions().dtype(torch::kLong).device(flat.device());
  auto token_indices =
      torch::arange(0, flat.size(0), long_options).repeat_interleave(config_.num_experts_per_tok).contiguous();
  auto expert_indices = topk_indices.reshape({route_count}).contiguous();
  auto route_weights = topk_weights.reshape({route_count}).contiguous();
  auto destination_ranks = torch::floor_divide(expert_indices, experts_per_rank).contiguous();
  auto route_tokens = flat.index_select(0, token_indices);
  auto dispatch = distributed::expert_parallel_dispatch_equal_capacity(
      route_tokens.contiguous(), destination_ranks, *expert_group.collectives, expert_group.ranks);

  const int64_t capacity = dispatch.capacity;
  auto metadata_size = expert_group.world_size * capacity;
  auto source_token_payload = torch::full({metadata_size}, -1, long_options);
  auto expert_payload = torch::full({metadata_size}, -1, long_options);
  auto weight_payload = torch::zeros({metadata_size}, route_weights.options());
  if (capacity > 0) {
    for (int64_t dest = 0; dest < expert_group.world_size; ++dest) {
      auto positions =
          torch::nonzero(destination_ranks == dest).flatten().to(torch::TensorOptions().dtype(torch::kLong).device(flat.device()));
      const int64_t count = positions.numel();
      if (count == 0) {
        continue;
      }
      auto payload_indices =
          torch::arange(dest * capacity, dest * capacity + count, torch::TensorOptions().dtype(torch::kLong).device(flat.device()));
      source_token_payload.index_put_({payload_indices}, token_indices.index_select(0, positions));
      expert_payload.index_put_({payload_indices}, expert_indices.index_select(0, positions));
      weight_payload = weight_payload.index_add(0, payload_indices, route_weights.index_select(0, positions));
    }
  }

  auto received_experts = expert_group.collectives->all_to_all(expert_payload.contiguous(), expert_group.ranks, 0)
                              .contiguous();
  auto local_return = torch::zeros_like(dispatch.received_tokens);
  int64_t local_expert_capacity = 0;
  std::vector<torch::Tensor> local_expert_positions;
  local_expert_positions.reserve(static_cast<size_t>(experts_per_rank));
  for (int64_t expert = expert_begin; expert < expert_end; ++expert) {
    auto positions = torch::nonzero(received_experts == expert).flatten().to(long_options);
    local_expert_capacity = std::max<int64_t>(local_expert_capacity, positions.numel());
    local_expert_positions.push_back(positions);
  }
  if (local_expert_capacity > 0) {
    std::vector<torch::Tensor> packed_indices_parts;
    std::vector<torch::Tensor> return_indices_parts;
    packed_indices_parts.reserve(static_cast<size_t>(experts_per_rank));
    return_indices_parts.reserve(static_cast<size_t>(experts_per_rank));
    for (int64_t local_expert = 0; local_expert < experts_per_rank; ++local_expert) {
      const auto& positions = local_expert_positions.at(static_cast<size_t>(local_expert));
      const int64_t count = positions.numel();
      if (count == 0) {
        continue;
      }
      packed_indices_parts.push_back(
          torch::arange(local_expert * local_expert_capacity,
                        local_expert * local_expert_capacity + count,
                        long_options));
      return_indices_parts.push_back(positions);
    }

    auto packed_indices = torch::cat(packed_indices_parts, 0).contiguous();
    auto return_indices = torch::cat(return_indices_parts, 0).contiguous();
    auto packed_flat = torch::zeros({experts_per_rank * local_expert_capacity, config_.hidden_size},
                                    dispatch.received_tokens.options());
    packed_flat = packed_flat.index_add(0, packed_indices, dispatch.received_tokens.index_select(0, return_indices));
    auto packed_tokens = packed_flat.view({experts_per_rank, local_expert_capacity, config_.hidden_size});

    std::vector<torch::Tensor> gate_weights;
    std::vector<torch::Tensor> up_weights;
    std::vector<torch::Tensor> down_weights;
    gate_weights.reserve(static_cast<size_t>(experts_per_rank));
    up_weights.reserve(static_cast<size_t>(experts_per_rank));
    down_weights.reserve(static_cast<size_t>(experts_per_rank));
    for (int64_t expert = expert_begin; expert < expert_end; ++expert) {
      std::string ep = p + "experts." + std::to_string(expert) + ".";
      gate_weights.push_back(local_or_shard_dim(weight(ep + "gate_proj.weight"),
                                                0,
                                                config_.moe_intermediate_size,
                                                tensor_group.rank,
                                                tensor_group.world_size,
                                                ep + "gate_proj.weight"));
      up_weights.push_back(local_or_shard_dim(weight(ep + "up_proj.weight"),
                                              0,
                                              config_.moe_intermediate_size,
                                              tensor_group.rank,
                                              tensor_group.world_size,
                                              ep + "up_proj.weight"));
      down_weights.push_back(local_or_shard_dim(weight(ep + "down_proj.weight"),
                                                1,
                                                config_.moe_intermediate_size,
                                                tensor_group.rank,
                                                tensor_group.world_size,
                                                ep + "down_proj.weight"));
    }

    auto gate_stack = torch::stack(gate_weights, 0).contiguous();
    auto up_stack = torch::stack(up_weights, 0).contiguous();
    auto down_stack = torch::stack(down_weights, 0).contiguous();
    if (gate_stack.size(1) != intermediate_local || up_stack.size(1) != intermediate_local ||
        down_stack.size(2) != intermediate_local) {
      throw std::invalid_argument("Qwen3 MoE TP expert shard shape mismatch");
    }
    auto gate = torch::bmm(packed_tokens, gate_stack.transpose(1, 2));
    auto up = torch::bmm(packed_tokens, up_stack.transpose(1, 2));
    auto expert_out = torch::bmm(torch::silu(gate) * up, down_stack.transpose(1, 2));
    auto expert_out_flat = expert_out.reshape({experts_per_rank * local_expert_capacity, config_.hidden_size});
    local_return = local_return.index_add(0, return_indices, expert_out_flat.index_select(0, packed_indices));
  }

  auto returned = distributed::expert_parallel_all_to_all_autograd(
      local_return.contiguous(), *expert_group.collectives, expert_group.ranks, 0);
  auto valid_positions = torch::nonzero(source_token_payload >= 0).flatten().to(long_options);
  auto out = torch::zeros_like(flat);
  if (valid_positions.numel() > 0) {
    auto weighted = returned.index_select(0, valid_positions) *
                    weight_payload.index_select(0, valid_positions).unsqueeze(-1).to(returned.scalar_type());
    out.index_add_(0, source_token_payload.index_select(0, valid_positions), weighted);
  }
  if (tensor_group.world_size > 1) {
    out = all_reduce_sum_preserve_local_grad(out.contiguous(), tensor_group);
  }
  return out.reshape_as(x);
}

torch::Tensor Qwen35TextModel::mlp_tensor_parallel(const torch::Tensor& x,
                                                   int64_t layer_idx,
                                                   const distributed::ParallelGroup& tensor_group) {
  if (config_.is_qwen3_moe()) {
    if (tensor_group.world_size != 1) {
      throw std::invalid_argument("Qwen3 MoE TP/EP sparse MLP is not implemented yet; use TP=1 with PP/DP for SFT smoke");
    }
    return mlp(x, layer_idx);
  }
  std::string p = layer_prefix(config_, layer_idx) + "mlp.";
  auto gate_weight = local_or_shard_dim(
      weight(p + "gate_proj.weight"), 0, config_.intermediate_size, tensor_group.rank, tensor_group.world_size,
      p + "gate_proj.weight");
  auto up_weight = local_or_shard_dim(
      weight(p + "up_proj.weight"), 0, config_.intermediate_size, tensor_group.rank, tensor_group.world_size,
      p + "up_proj.weight");
  auto down_weight = local_or_shard_dim(
      weight(p + "down_proj.weight"), 1, config_.intermediate_size, tensor_group.rank, tensor_group.world_size,
      p + "down_proj.weight");
  auto gate = dense(x, gate_weight);
  auto up = dense(x, up_weight);
  auto partial = dense(torch::silu(gate) * up, down_weight);
  if (tensor_group.world_size == 1) {
    return partial;
  }
  if (tensor_group.collectives == nullptr) {
    throw std::invalid_argument("Qwen3.5 MLP TP requires collectives");
  }
  return all_reduce_sum_preserve_local_grad(partial.contiguous(), tensor_group);
}

torch::Tensor Qwen35TextModel::mlp_tensor_expert_parallel(const torch::Tensor& x,
                                                          int64_t layer_idx,
                                                          const distributed::ParallelGroup& tensor_group,
                                                          const distributed::ParallelGroup& expert_group) {
  if (config_.is_qwen3_moe()) {
    return mlp_expert_parallel(x, layer_idx, tensor_group, expert_group);
  }
  return mlp_tensor_parallel(x, layer_idx, tensor_group);
}

torch::Tensor Qwen35TextModel::full_attention(const torch::Tensor& x, int64_t layer_idx) {
  std::string p = layer_prefix(config_, layer_idx) + "self_attn.";
  int64_t b = x.size(0);
  int64_t s = x.size(1);
  int64_t h = config_.num_attention_heads;
  int64_t kvh = config_.num_key_value_heads;
  int64_t d = config_.head_dim;

  torch::Tensor q;
  torch::Tensor gate;
  if (config_.is_qwen3_moe()) {
    q = rms_norm(dense(x, weight(p + "q_proj.weight")).view({b, s, h, d}), weight(p + "q_norm.weight")).transpose(1, 2);
  } else {
    auto qg = dense(x, weight(p + "q_proj.weight")).view({b, s, h, d * 2});
    auto chunks = qg.chunk(2, -1);
    q = rms_norm(chunks[0], weight(p + "q_norm.weight")).transpose(1, 2);
    gate = chunks[1].reshape({b, s, h * d});
  }
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
  auto mask = causal_mask(s, x.device());
  attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1.0e9);
  attn = torch::softmax(attn.to(torch::kFloat32), -1);
  auto out = torch::matmul(attn, v.to(torch::kFloat32)).transpose(1, 2).contiguous().reshape({b, s, h * d});
  if (!config_.is_qwen3_moe()) {
    out = out * torch::sigmoid(gate.to(torch::kFloat32));
  }
  out = out.to(x.scalar_type());
  return dense(out, weight(p + "o_proj.weight"));
}

torch::Tensor Qwen35TextModel::full_attention_context_parallel(const torch::Tensor& x,
                                                               int64_t layer_idx,
                                                               const distributed::ParallelGroup& context_group,
                                                               int64_t original_sequence_length) {
  const int64_t context_rank = group_local_rank(context_group, "Qwen3.5 CP full attention");
  if (context_group.world_size == 1) {
    return full_attention(x, layer_idx);
  }
  if (context_group.world_size > 1 && context_group.collectives == nullptr) {
    throw std::invalid_argument("Qwen3.5 CP full attention requires collectives");
  }
  std::string p = layer_prefix(config_, layer_idx) + "self_attn.";
  int64_t b = x.size(0);
  int64_t s_local = x.size(1);
  int64_t h = config_.num_attention_heads;
  int64_t kvh = config_.num_key_value_heads;
  int64_t d = config_.head_dim;
  const int64_t query_begin = context_rank * s_local;

  torch::Tensor q;
  torch::Tensor gate;
  if (config_.is_qwen3_moe()) {
    q = rms_norm(dense(x, weight(p + "q_proj.weight")).view({b, s_local, h, d}), weight(p + "q_norm.weight"))
            .transpose(1, 2);
  } else {
    auto qg = dense(x, weight(p + "q_proj.weight")).view({b, s_local, h, d * 2});
    auto chunks = qg.chunk(2, -1);
    q = rms_norm(chunks[0], weight(p + "q_norm.weight")).transpose(1, 2);
    gate = chunks[1].reshape({b, s_local, h * d});
  }
  auto k = rms_norm(dense(x, weight(p + "k_proj.weight")).view({b, s_local, kvh, d}), weight(p + "k_norm.weight"))
               .transpose(1, 2);
  auto v = dense(x, weight(p + "v_proj.weight")).view({b, s_local, kvh, d}).transpose(1, 2);

  auto rope = rotary_embeddings_range(b, query_begin, s_local, x.device(), x.scalar_type());
  auto cos = rope.first.unsqueeze(1);
  auto sin = rope.second.unsqueeze(1);
  int64_t rotary_dim = cos.size(-1);
  auto q_rot = q.index({Slice(), Slice(), Slice(), Slice(0, rotary_dim)});
  auto q_pass = q.index({Slice(), Slice(), Slice(), Slice(rotary_dim, None)});
  auto k_rot = k.index({Slice(), Slice(), Slice(), Slice(0, rotary_dim)});
  auto k_pass = k.index({Slice(), Slice(), Slice(), Slice(rotary_dim, None)});
  q = torch::cat(std::vector<torch::Tensor>{q_rot * cos + rotate_half(q_rot) * sin, q_pass}, -1);
  k = torch::cat(std::vector<torch::Tensor>{k_rot * cos + rotate_half(k_rot) * sin, k_pass}, -1);

  k = repeat_kv(k, h / kvh).contiguous();
  v = repeat_kv(v, h / kvh).contiguous();
  auto context_out = distributed::context_parallel_causal_attention_ring_exchange_kv(
      q.contiguous(),
      k,
      v,
      *context_group.collectives,
      context_group.ranks,
      context_rank,
      original_sequence_length,
      1.0 / std::sqrt(static_cast<double>(d)));
  auto out = context_out.transpose(1, 2).contiguous().reshape({b, s_local, h * d});
  if (!config_.is_qwen3_moe()) {
    out = out * torch::sigmoid(gate.to(torch::kFloat32));
  }
  out = out.to(x.scalar_type());
  return dense(out, weight(p + "o_proj.weight"));
}

torch::Tensor Qwen35TextModel::full_attention_tensor_parallel(const torch::Tensor& x,
                                                              int64_t layer_idx,
                                                              const distributed::ParallelGroup& tensor_group) {
  if (config_.is_qwen3_moe() && tensor_group.world_size == 1) {
    return full_attention(x, layer_idx);
  }
  std::string p = layer_prefix(config_, layer_idx) + "self_attn.";
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
  int64_t kvh_local = kvh / tensor_group.world_size;
  int64_t d = config_.head_dim;

  torch::Tensor q;
  torch::Tensor gate;
  if (config_.is_qwen3_moe()) {
    auto q_weight = local_or_shard_dim(
        weight(p + "q_proj.weight"), 0, h * d, tensor_group.rank, tensor_group.world_size, p + "q_proj.weight");
    q = rms_norm(dense(x, q_weight).view({b, s, h_local, d}), weight(p + "q_norm.weight")).transpose(1, 2);
  } else {
    auto q_weight = local_or_shard_dim(
        weight(p + "q_proj.weight"), 0, h * d * 2, tensor_group.rank, tensor_group.world_size, p + "q_proj.weight");
    auto qg = dense(x, q_weight).view({b, s, h_local, d * 2});
    auto chunks = qg.chunk(2, -1);
    q = rms_norm(chunks[0], weight(p + "q_norm.weight")).transpose(1, 2);
    gate = chunks[1].reshape({b, s, h_local * d});
  }

  auto k_weight = local_or_shard_dim(
      weight(p + "k_proj.weight"), 0, kvh * d, tensor_group.rank, tensor_group.world_size, p + "k_proj.weight");
  auto v_weight = local_or_shard_dim(
      weight(p + "v_proj.weight"), 0, kvh * d, tensor_group.rank, tensor_group.world_size, p + "v_proj.weight");
  auto k = rms_norm(dense(x, k_weight).view({b, s, kvh_local, d}), weight(p + "k_norm.weight")).transpose(1, 2);
  auto v = dense(x, v_weight).view({b, s, kvh_local, d}).transpose(1, 2);

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

  k = repeat_kv(k, h / kvh).contiguous();
  v = repeat_kv(v, h / kvh).contiguous();
  auto attn = torch::matmul(q, k.transpose(2, 3)) * (1.0 / std::sqrt(static_cast<double>(d)));
  auto mask = causal_mask(s, x.device());
  attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1.0e9);
  attn = torch::softmax(attn.to(torch::kFloat32), -1);
  auto local = torch::matmul(attn, v.to(torch::kFloat32)).transpose(1, 2).contiguous().reshape({b, s, h_local * d});
  if (!config_.is_qwen3_moe()) {
    local = local * torch::sigmoid(gate.to(torch::kFloat32));
  }
  local = local.to(x.scalar_type());
  auto o_weight_shard = local_or_shard_dim(
      weight(p + "o_proj.weight"), 1, h * d, tensor_group.rank, tensor_group.world_size, p + "o_proj.weight");
  auto partial = dense(local, o_weight_shard);
  if (tensor_group.world_size == 1) {
    return partial;
  }
  if (tensor_group.collectives == nullptr) {
    throw std::invalid_argument("Qwen3.5 full attention TP requires collectives");
  }
  return all_reduce_sum_preserve_local_grad(partial.contiguous(), tensor_group);
}

torch::Tensor Qwen35TextModel::linear_attention(const torch::Tensor& x, int64_t layer_idx) {
  std::string p = layer_prefix(config_, layer_idx) + "linear_attn.";
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

  auto core = linear_attention_recurrent(query, key, value, beta, g, kd, vd)
                  .transpose(1, 2)
                  .contiguous()
                  .reshape({bsz * seq * vh, vd});
  auto zg = z.reshape({bsz * seq * vh, vd});
  core = rms_norm_gated(core, zg, weight(p + "norm.weight")).reshape({bsz, seq, value_dim});
  return dense(core, weight(p + "out_proj.weight"));
}

torch::Tensor Qwen35TextModel::linear_attention_context_parallel(const torch::Tensor& x,
                                                                 int64_t layer_idx,
                                                                 const distributed::ParallelGroup& context_group,
                                                                 int64_t original_sequence_length) {
  const int64_t context_rank = group_local_rank(context_group, "Qwen3.5 CP linear attention");
  if (context_group.world_size == 1) {
    return linear_attention(x, layer_idx);
  }
  if (context_group.collectives == nullptr) {
    throw std::invalid_argument("Qwen3.5 CP linear attention requires collectives");
  }
  std::string p = layer_prefix(config_, layer_idx) + "linear_attn.";
  int64_t bsz = x.size(0);
  int64_t seq_local = x.size(1);
  int64_t kh = config_.linear_num_key_heads;
  int64_t vh = config_.linear_num_value_heads;
  int64_t kd = config_.linear_key_head_dim;
  int64_t vd = config_.linear_value_head_dim;
  int64_t key_dim = kh * kd;
  int64_t value_dim = vh * vd;
  int64_t conv_dim = key_dim * 2 + value_dim;

  auto mixed_local = dense(x, weight(p + "in_proj_qkv.weight"));
  auto mixed_global = ring_exchange_rank_order(mixed_local, context_group, 1).narrow(1, 0, original_sequence_length);
  auto mixed = mixed_global.transpose(1, 2);
  auto conv_opts = torch::nn::functional::Conv1dFuncOptions().padding(config_.linear_conv_kernel_dim - 1).groups(conv_dim);
  mixed = torch::nn::functional::conv1d(mixed, weight(p + "conv1d.weight"), conv_opts);
  mixed = torch::silu(mixed.index({Slice(), Slice(), Slice(0, original_sequence_length)})).transpose(1, 2);

  auto qkv = mixed.split({key_dim, key_dim, value_dim}, -1);
  auto query = qkv[0].reshape({bsz, original_sequence_length, kh, kd});
  auto key = qkv[1].reshape({bsz, original_sequence_length, kh, kd});
  auto value = qkv[2].reshape({bsz, original_sequence_length, vh, vd});
  auto z_local = dense(x, weight(p + "in_proj_z.weight")).reshape({bsz, seq_local, vh, vd});
  auto z = ring_exchange_rank_order(z_local, context_group, 1).narrow(1, 0, original_sequence_length);
  auto beta_local = torch::sigmoid(dense(x, weight(p + "in_proj_b.weight")));
  auto beta = ring_exchange_rank_order(beta_local, context_group, 1).narrow(1, 0, original_sequence_length);
  auto a_local = dense(x, weight(p + "in_proj_a.weight"));
  auto a = ring_exchange_rank_order(a_local, context_group, 1).narrow(1, 0, original_sequence_length);
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

  auto query_local = distributed::context_parallel_slice_padded(query, context_rank, context_group.world_size, 2, 0.0);
  auto key_local = distributed::context_parallel_slice_padded(key, context_rank, context_group.world_size, 2, 0.0);
  auto value_local = distributed::context_parallel_slice_padded(value, context_rank, context_group.world_size, 2, 0.0);
  auto beta_local_scan =
      distributed::context_parallel_slice_padded(beta, context_rank, context_group.world_size, 2, 0.0);
  auto g_local = distributed::context_parallel_slice_padded(g, context_rank, context_group.world_size, 2, 0.0);
  auto state_like = torch::empty({bsz, vh, kd, vd}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
  auto initial_state = linear_attention_context_initial_state(state_like, context_group, context_rank);
  auto scan = linear_attention_recurrent_with_state(query_local.contiguous(),
                                                    key_local.contiguous(),
                                                    value_local.contiguous(),
                                                    beta_local_scan.contiguous(),
                                                    g_local.contiguous(),
                                                    initial_state,
                                                    kd,
                                                    vd);
  auto scan_output =
      linear_attention_attach_final_state_carry(std::get<0>(scan), std::get<1>(scan), context_group, context_rank);
  auto core_local = scan_output
                        .transpose(1, 2)
                        .contiguous()
                        .reshape({bsz * seq_local * vh, vd});
  auto zg = distributed::context_parallel_slice_padded(z, context_rank, context_group.world_size, 1, 0.0)
                .reshape({bsz * seq_local * vh, vd});
  auto core = rms_norm_gated(core_local, zg, weight(p + "norm.weight")).reshape({bsz, seq_local, value_dim});
  return dense(core, weight(p + "out_proj.weight"));
}

torch::Tensor Qwen35TextModel::linear_attention_tensor_parallel(const torch::Tensor& x,
                                                                int64_t layer_idx,
                                                                const distributed::ParallelGroup& tensor_group) {
  std::string p = layer_prefix(config_, layer_idx) + "linear_attn.";
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

  auto qkv_full_or_local = weight(p + "in_proj_qkv.weight");
  torch::Tensor qkv_weight;
  if (qkv_full_or_local.size(0) == conv_local_dim) {
    qkv_weight = qkv_full_or_local;
  } else {
    auto q_weight = qkv_full_or_local.narrow(0, key_begin, key_local_dim);
    auto k_weight = qkv_full_or_local.narrow(0, key_dim + key_begin, key_local_dim);
    auto v_weight = qkv_full_or_local.narrow(0, key_dim * 2 + value_begin, value_local_dim);
    qkv_weight = torch::cat(std::vector<torch::Tensor>{q_weight, k_weight, v_weight}, 0).contiguous();
  }
  auto mixed = dense(x, qkv_weight).transpose(1, 2);

  auto conv_full_or_local = weight(p + "conv1d.weight");
  torch::Tensor conv_weight;
  if (conv_full_or_local.size(0) == conv_local_dim) {
    conv_weight = conv_full_or_local;
  } else {
    auto conv_q = conv_full_or_local.narrow(0, key_begin, key_local_dim);
    auto conv_k = conv_full_or_local.narrow(0, key_dim + key_begin, key_local_dim);
    auto conv_v = conv_full_or_local.narrow(0, key_dim * 2 + value_begin, value_local_dim);
    conv_weight = torch::cat(std::vector<torch::Tensor>{conv_q, conv_k, conv_v}, 0).contiguous();
  }
  auto conv_opts = torch::nn::functional::Conv1dFuncOptions().padding(config_.linear_conv_kernel_dim - 1).groups(conv_local_dim);
  mixed = torch::nn::functional::conv1d(mixed, conv_weight, conv_opts);
  mixed = torch::silu(mixed.index({Slice(), Slice(), Slice(0, seq)})).transpose(1, 2);

  auto qkv = mixed.split({key_local_dim, key_local_dim, value_local_dim}, -1);
  auto query = qkv[0].reshape({bsz, seq, kh_local, kd});
  auto key = qkv[1].reshape({bsz, seq, kh_local, kd});
  auto value = qkv[2].reshape({bsz, seq, vh_local, vd});
  auto z_weight = local_or_shard_dim(
      weight(p + "in_proj_z.weight"), 0, value_dim, tensor_group.rank, tensor_group.world_size, p + "in_proj_z.weight");
  auto b_weight = local_or_shard_dim(
      weight(p + "in_proj_b.weight"), 0, vh, tensor_group.rank, tensor_group.world_size, p + "in_proj_b.weight");
  auto a_weight = local_or_shard_dim(
      weight(p + "in_proj_a.weight"), 0, vh, tensor_group.rank, tensor_group.world_size, p + "in_proj_a.weight");
  auto a_log = local_or_shard_dim(
      weight(p + "A_log"), 0, vh, tensor_group.rank, tensor_group.world_size, p + "A_log");
  auto dt_bias = local_or_shard_dim(
      weight(p + "dt_bias"), 0, vh, tensor_group.rank, tensor_group.world_size, p + "dt_bias");
  auto z = dense(x, z_weight).reshape({bsz, seq, vh_local, vd});
  auto beta = torch::sigmoid(dense(x, b_weight));
  auto a = dense(x, a_weight);
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

  auto core = linear_attention_recurrent(query, key, value, beta, g, kd, vd)
                  .transpose(1, 2)
                  .contiguous()
                  .reshape({bsz * seq * vh_local, vd});
  auto zg = z.reshape({bsz * seq * vh_local, vd});
  core = rms_norm_gated(core, zg, weight(p + "norm.weight")).reshape({bsz, seq, value_local_dim});
  auto out_weight_shard = local_or_shard_dim(
      weight(p + "out_proj.weight"), 1, value_dim, tensor_group.rank, tensor_group.world_size, p + "out_proj.weight");
  auto partial = dense(core, out_weight_shard);
  if (tensor_group.world_size == 1) {
    return partial;
  }
  if (tensor_group.collectives == nullptr) {
    throw std::invalid_argument("Qwen3.5 linear attention TP requires collectives");
  }
  return all_reduce_sum_preserve_local_grad(partial.contiguous(), tensor_group);
}

torch::Tensor Qwen35TextModel::forward_hidden(const torch::Tensor& input_ids, int64_t max_layers) {
  auto hidden = embed(input_ids);
  int64_t layers = max_layers < 0 ? config_.num_hidden_layers : std::min(max_layers, config_.num_hidden_layers);
  for (int64_t i = 0; i < layers; ++i) {
    std::string p = layer_prefix(config_, i);
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
  return rms_norm(hidden, weight(final_norm_weight_name(config_)));
}

torch::Tensor Qwen35TextModel::forward_hidden_context_parallel(const torch::Tensor& input_ids,
                                                               const distributed::ParallelGroup& context_group,
                                                               int64_t max_layers) {
  const int64_t context_rank = group_local_rank(context_group, "Qwen3.5 CP forward");
  if (context_group.world_size == 1) {
    return forward_hidden(input_ids, max_layers);
  }
  auto hidden = distributed::context_parallel_slice_padded(
      embed(input_ids), context_rank, context_group.world_size, 1, 0.0);
  int64_t layers = max_layers < 0 ? config_.num_hidden_layers : std::min(max_layers, config_.num_hidden_layers);
  return forward_hidden_range_context_parallel(hidden,
                                               0,
                                               layers,
                                               context_group,
                                               input_ids.size(1),
                                               true);
}

torch::Tensor Qwen35TextModel::forward_hidden_range_context_parallel(const torch::Tensor& hidden_local,
                                                                     int64_t layer_begin,
                                                                     int64_t layer_end,
                                                                     const distributed::ParallelGroup& context_group,
                                                                     int64_t original_sequence_length,
                                                                     bool apply_final_norm) {
  distributed::ParallelGroup expert_group;
  return forward_hidden_range_context_expert_parallel(hidden_local,
                                                      layer_begin,
                                                      layer_end,
                                                      context_group,
                                                      expert_group,
                                                      original_sequence_length,
                                                      apply_final_norm);
}

torch::Tensor Qwen35TextModel::forward_hidden_range_context_expert_parallel(
    const torch::Tensor& hidden_local,
    int64_t layer_begin,
    int64_t layer_end,
    const distributed::ParallelGroup& context_group,
    const distributed::ParallelGroup& expert_group,
    int64_t original_sequence_length,
    bool apply_final_norm) {
  if (layer_begin < 0 || layer_end < layer_begin || layer_end > config_.num_hidden_layers) {
    throw std::invalid_argument("invalid Qwen3.5 CP layer range");
  }
  (void)group_local_rank(context_group, "Qwen3.5 CP layer range");
  if (context_group.world_size == 1) {
    auto out = hidden_local;
    for (int64_t i = layer_begin; i < layer_end; ++i) {
      std::string p = layer_prefix(config_, i);
      auto residual = out;
      out = rms_norm(out, weight(p + "input_layernorm.weight"));
      if (config_.layer_types.at(static_cast<size_t>(i)) == "full_attention") {
        out = full_attention(out, i);
      } else {
        out = linear_attention(out, i);
      }
      out = residual + out;
      residual = out;
      out = rms_norm(out, weight(p + "post_attention_layernorm.weight"));
      out = residual + mlp_tensor_expert_parallel(out, i, distributed::ParallelGroup{}, expert_group);
    }
    return apply_final_norm ? rms_norm(out, weight(final_norm_weight_name(config_))) : out;
  }
  if (context_group.collectives == nullptr) {
    throw std::invalid_argument("Qwen3.5 CP layer range requires collectives");
  }
  auto out = hidden_local;
  for (int64_t i = layer_begin; i < layer_end; ++i) {
    std::string p = layer_prefix(config_, i);
    auto residual = out;
    out = rms_norm(out, weight(p + "input_layernorm.weight"));
    if (config_.layer_types.at(static_cast<size_t>(i)) == "full_attention") {
      out = full_attention_context_parallel(out, i, context_group, original_sequence_length);
    } else {
      out = linear_attention_context_parallel(out, i, context_group, original_sequence_length);
    }
    out = residual + out;
    residual = out;
    out = rms_norm(out, weight(p + "post_attention_layernorm.weight"));
    out = residual + mlp_tensor_expert_parallel(out, i, distributed::ParallelGroup{}, expert_group);
  }
  if (apply_final_norm) {
    out = rms_norm(out, weight(final_norm_weight_name(config_)));
  }
  return out;
}

torch::Tensor Qwen35TextModel::token_embeddings(const torch::Tensor& input_ids) {
  return embed(input_ids);
}

torch::Tensor Qwen35TextModel::token_embeddings_tensor_parallel(const torch::Tensor& input_ids,
                                                                const distributed::ParallelGroup& tensor_group) {
  auto emb = weight(embed_weight_name(config_));
  if (tensor_group.world_size == 1) {
    return embed(input_ids);
  }
  if (tensor_group.collectives == nullptr) {
    throw std::invalid_argument("Qwen3.5 vocab-parallel embedding requires collectives");
  }
  if (config_.vocab_size % tensor_group.world_size != 0) {
    throw std::invalid_argument("Qwen3.5 vocab size must be divisible by TP size");
  }
  const int64_t vocab_local = config_.vocab_size / tensor_group.world_size;
  if (emb.size(0) != vocab_local) {
    emb = emb.narrow(0, tensor_group.rank * vocab_local, vocab_local).contiguous();
  }
  const int64_t start = tensor_group.rank * vocab_local;
  const int64_t end = start + vocab_local;
  auto ids = input_ids.to(torch::TensorOptions().dtype(torch::kLong).device(emb.device()));
  auto mask = (ids >= start) * (ids < end);
  auto local_ids = (ids - start).clamp(0, vocab_local - 1);
  auto local = torch::nn::functional::embedding(local_ids, emb);
  local = local * mask.unsqueeze(-1).to(local.scalar_type());
  return all_reduce_sum_preserve_local_grad(local.contiguous(), tensor_group);
}

torch::Tensor Qwen35TextModel::lm_head_logits(const torch::Tensor& hidden) {
  return torch::matmul(hidden.to(torch::kFloat32), weight(lm_head_weight_name(config_)).to(torch::kFloat32).transpose(0, 1));
}

torch::Tensor Qwen35TextModel::response_log_probs_tensor_parallel(const torch::Tensor& hidden,
                                                                  const torch::Tensor& response_ids,
                                                                  const distributed::ParallelGroup& tensor_group) {
  auto emb = weight(lm_head_weight_name(config_));
  if (tensor_group.world_size == 1) {
    return torch::log_softmax(torch::matmul(hidden.to(torch::kFloat32), emb.to(torch::kFloat32).transpose(0, 1)), -1)
        .gather(-1, response_ids.unsqueeze(-1))
        .squeeze(-1);
  }
  if (tensor_group.collectives == nullptr) {
    throw std::invalid_argument("Qwen3.5 vocab-parallel logprob requires collectives");
  }
  if (config_.vocab_size % tensor_group.world_size != 0) {
    throw std::invalid_argument("Qwen3.5 vocab size must be divisible by TP size");
  }
  const int64_t vocab_local = config_.vocab_size / tensor_group.world_size;
  if (emb.size(0) != vocab_local) {
    emb = emb.narrow(0, tensor_group.rank * vocab_local, vocab_local).contiguous();
  }
  auto local_logits = torch::matmul(hidden.to(torch::kFloat32), emb.to(torch::kFloat32).transpose(0, 1));
  auto local_max = std::get<0>(local_logits.max(-1, true)).contiguous();
  auto global_max =
      tensor_group.collectives->all_reduce(local_max, distributed::ReduceOp::Max, tensor_group.ranks);
  auto local_exp_sum = torch::exp(local_logits - global_max).sum(-1, true).contiguous();
  auto global_exp_sum = all_reduce_sum_preserve_local_grad(local_exp_sum, tensor_group);

  const int64_t start = tensor_group.rank * vocab_local;
  const int64_t end = start + vocab_local;
  auto ids = response_ids.to(torch::TensorOptions().dtype(torch::kLong).device(hidden.device()));
  auto mask = (ids >= start) * (ids < end);
  auto local_ids = (ids - start).clamp(0, vocab_local - 1);
  auto local_target = local_logits.gather(-1, local_ids.unsqueeze(-1)).squeeze(-1);
  local_target = local_target * mask.to(local_target.scalar_type());
  auto target = all_reduce_sum_preserve_local_grad(local_target.contiguous(), tensor_group);
  return target - global_max.squeeze(-1) - torch::log(global_exp_sum.squeeze(-1));
}

torch::Tensor Qwen35TextModel::forward_hidden_tensor_parallel(const torch::Tensor& input_ids,
                                                              const distributed::ParallelGroup& tensor_group,
                                                              int64_t max_layers) {
  auto hidden = token_embeddings_tensor_parallel(input_ids, tensor_group);
  int64_t layers = max_layers < 0 ? config_.num_hidden_layers : std::min(max_layers, config_.num_hidden_layers);
  return forward_hidden_range_tensor_parallel(hidden, 0, layers, tensor_group, true);
}

torch::Tensor Qwen35TextModel::forward_hidden_range_tensor_parallel(const torch::Tensor& hidden,
                                                                    int64_t layer_begin,
                                                                    int64_t layer_end,
                                                                    const distributed::ParallelGroup& tensor_group,
                                                                    bool apply_final_norm) {
  distributed::ParallelGroup expert_group;
  return forward_hidden_range_tensor_expert_parallel(
      hidden, layer_begin, layer_end, tensor_group, expert_group, apply_final_norm);
}

torch::Tensor Qwen35TextModel::forward_hidden_range_tensor_expert_parallel(
    const torch::Tensor& hidden,
    int64_t layer_begin,
    int64_t layer_end,
    const distributed::ParallelGroup& tensor_group,
    const distributed::ParallelGroup& expert_group,
    bool apply_final_norm) {
  if (layer_begin < 0 || layer_end < layer_begin || layer_end > config_.num_hidden_layers) {
    throw std::invalid_argument("invalid Qwen3.5 layer range");
  }
  auto out = hidden;
  for (int64_t i = layer_begin; i < layer_end; ++i) {
    std::string p = layer_prefix(config_, i);
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
    out = residual + mlp_tensor_expert_parallel(out, i, tensor_group, expert_group);
  }
  if (apply_final_norm) {
    out = rms_norm(out, weight(final_norm_weight_name(config_)));
  }
  return out;
}

torch::Tensor Qwen35TextModel::forward_logits(const torch::Tensor& input_ids, int64_t max_layers) {
  auto hidden = forward_hidden(input_ids, max_layers);
  return lm_head_logits(hidden);
}

}  // namespace cverl
