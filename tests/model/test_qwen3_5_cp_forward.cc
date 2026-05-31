#include "cverl/distributed/collectives.h"
#include "cverl/distributed/context_parallel.h"
#include "cverl/distributed/parallel_ops.h"
#include "cverl/model/hf_model_loader.h"
#include "cverl/model/qwen3_5_text.h"

#include <torch/torch.h>
#include <torch/nn/functional/conv.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <unistd.h>

namespace {

void require(bool cond, const std::string& msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

class SendRecvCollectives final : public cverl::distributed::Collectives {
 public:
  SendRecvCollectives(int64_t rank,
                      int64_t world_size,
                      std::vector<torch::Tensor> responses,
                      std::vector<torch::Tensor> recv_responses = {})
      : responses_(std::move(responses)),
        recv_responses_(std::move(recv_responses)),
        rank_(rank),
        world_size_(world_size) {}

  int64_t rank() const override { return rank_; }
  int64_t world_size() const override { return world_size_; }
  void barrier() override {}
  torch::Tensor broadcast(const torch::Tensor& input, int64_t, const std::vector<int64_t>&) override { return input; }
  torch::Tensor all_reduce(const torch::Tensor& input,
                           cverl::distributed::ReduceOp,
                           const std::vector<int64_t>&) override {
    return input;
  }
  torch::Tensor all_gather(const torch::Tensor& input, const std::vector<int64_t>&, int64_t) override { return input; }
  torch::Tensor reduce_scatter(const torch::Tensor& input,
                               cverl::distributed::ReduceOp,
                               const std::vector<int64_t>&,
                               int64_t) override {
    const int64_t shard = input.size(0) / world_size_;
    return input.narrow(0, rank_ * shard, shard).contiguous();
  }
  void send(const torch::Tensor&, int64_t) override { ++send_calls_; }
  torch::Tensor recv_like(const torch::Tensor& like, int64_t) override {
    if (recv_next_ >= recv_responses_.size()) {
      throw std::runtime_error("unexpected recv_like in Qwen CP forward test");
    }
    auto out = recv_responses_.at(recv_next_++);
    require(out.sizes() == like.sizes(), "recv_like response shape mismatch");
    return out;
  }
  torch::Tensor send_recv(const torch::Tensor&, int64_t, const torch::Tensor& like, int64_t) override {
    if (next_ >= responses_.size()) {
      throw std::runtime_error("unexpected send_recv in Qwen CP forward test");
    }
    auto out = responses_.at(next_++);
    require(out.sizes() == like.sizes(), "send_recv response shape mismatch");
    return out;
  }

  int64_t send_recv_calls() const { return static_cast<int64_t>(next_); }
  int64_t send_calls() const { return send_calls_; }
  int64_t recv_calls() const { return static_cast<int64_t>(recv_next_); }

 private:
  std::vector<torch::Tensor> responses_;
  std::vector<torch::Tensor> recv_responses_;
  int64_t rank_ = 0;
  int64_t world_size_ = 1;
  size_t next_ = 0;
  size_t recv_next_ = 0;
  int64_t send_calls_ = 0;
};

torch::Tensor dense(const torch::Tensor& x, const torch::Tensor& w) {
  return torch::matmul(x, w.transpose(0, 1));
}

torch::Tensor rms_norm(const torch::Tensor& x, const torch::Tensor& w, double eps) {
  auto xf = x.to(torch::kFloat32);
  return (xf * torch::rsqrt((xf * xf).mean(-1, true) + eps) * (1.0 + w.to(torch::kFloat32))).to(x.scalar_type());
}

torch::Tensor l2norm(const torch::Tensor& x) {
  return x * torch::rsqrt((x * x).sum(-1, true) + 1e-6);
}

torch::Tensor rotate_half(const torch::Tensor& x) {
  const int64_t half = x.size(-1) / 2;
  auto x1 = x.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(),
                     torch::indexing::Slice(0, half)});
  auto x2 = x.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(),
                     torch::indexing::Slice(half, torch::indexing::None)});
  return torch::cat({-x2, x1}, -1);
}

std::pair<torch::Tensor, torch::Tensor> rotary(int64_t batch,
                                               int64_t begin,
                                               int64_t seq,
                                               int64_t rotary_dim,
                                               double theta) {
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  auto inv_idx = torch::arange(0, rotary_dim, 2, opts);
  auto inv_freq = torch::pow(theta, -(inv_idx / static_cast<double>(rotary_dim)));
  auto pos = torch::arange(begin, begin + seq, opts);
  auto freqs = torch::ger(pos, inv_freq);
  auto emb = torch::cat({freqs, freqs}, -1).unsqueeze(0).expand({batch, seq, rotary_dim});
  return {torch::cos(emb).contiguous(), torch::sin(emb).contiguous()};
}

torch::Tensor repeat_kv(const torch::Tensor& x, int64_t n_rep) {
  if (n_rep == 1) {
    return x;
  }
  auto sizes = x.sizes();
  return x.unsqueeze(2)
      .expand({sizes[0], sizes[1], n_rep, sizes[2], sizes[3]})
      .reshape({sizes[0], sizes[1] * n_rep, sizes[2], sizes[3]});
}

std::pair<torch::Tensor, torch::Tensor> remote_full_attention_kv(
    const torch::Tensor& hidden_local,
    int64_t position_begin,
    const std::unordered_map<std::string, torch::Tensor>& weights,
    const cverl::Qwen35TextConfig& cfg) {
  const std::string p = "model.language_model.layers.0.";
  auto x = rms_norm(hidden_local, weights.at(p + "input_layernorm.weight"), cfg.rms_norm_eps);
  const int64_t b = x.size(0);
  const int64_t s = x.size(1);
  auto k = rms_norm(dense(x, weights.at(p + "self_attn.k_proj.weight")).view({b, s, cfg.num_key_value_heads, cfg.head_dim}),
                    weights.at(p + "self_attn.k_norm.weight"),
                    cfg.rms_norm_eps)
               .transpose(1, 2);
  auto v = dense(x, weights.at(p + "self_attn.v_proj.weight")).view({b, s, cfg.num_key_value_heads, cfg.head_dim})
               .transpose(1, 2);
  const int64_t rotary_dim = static_cast<int64_t>(cfg.head_dim * cfg.partial_rotary_factor);
  auto rope = rotary(b, position_begin, s, rotary_dim, cfg.rope_theta);
  auto cos = rope.first.unsqueeze(1);
  auto sin = rope.second.unsqueeze(1);
  auto k_rot = k.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(),
                        torch::indexing::Slice(0, rotary_dim)});
  auto k_pass = k.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(),
                         torch::indexing::Slice(rotary_dim, torch::indexing::None)});
  k = torch::cat({k_rot * cos + rotate_half(k_rot) * sin, k_pass}, -1);
  return {repeat_kv(k, cfg.num_attention_heads / cfg.num_key_value_heads).contiguous(),
          repeat_kv(v, cfg.num_attention_heads / cfg.num_key_value_heads).contiguous()};
}

std::filesystem::path make_model_dir(const std::string& layer_type) {
  auto dir = std::filesystem::temp_directory_path() /
             ("cverl_qwen_cp_forward_" + layer_type + "_" + std::to_string(::getpid()));
  std::filesystem::create_directories(dir);
  std::ofstream config(dir / "config.json");
  config << R"({
    "model_type": "qwen3_5",
    "architectures": ["Qwen3ForCausalLM"],
    "dtype": "float32",
    "vocab_size": 32,
    "hidden_size": 4,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 2,
    "intermediate_size": 6,
    "linear_key_head_dim": 2,
    "linear_value_head_dim": 2,
    "linear_num_key_heads": 1,
    "linear_num_value_heads": 1,
    "linear_conv_kernel_dim": 2,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "partial_rotary_factor": 1.0,
    "layer_types": [")" << layer_type << R"("]
  })";
  return dir;
}

std::unordered_map<std::string, torch::Tensor> make_weights(const cverl::Qwen35TextConfig& cfg) {
  torch::manual_seed(23);
  std::unordered_map<std::string, torch::Tensor> w;
  auto small = [](std::vector<int64_t> shape) { return torch::randn(shape, torch::kFloat32) * 0.05; };
  w["model.language_model.embed_tokens.weight"] = small({cfg.vocab_size, cfg.hidden_size});
  w["model.language_model.norm.weight"] = torch::zeros({cfg.hidden_size}, torch::kFloat32);
  const std::string p = "model.language_model.layers.0.";
  w[p + "input_layernorm.weight"] = torch::zeros({cfg.hidden_size}, torch::kFloat32);
  w[p + "post_attention_layernorm.weight"] = torch::zeros({cfg.hidden_size}, torch::kFloat32);
  w[p + "self_attn.q_proj.weight"] = small({cfg.num_attention_heads * cfg.head_dim * 2, cfg.hidden_size});
  w[p + "self_attn.k_proj.weight"] = small({cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size});
  w[p + "self_attn.v_proj.weight"] = small({cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size});
  w[p + "self_attn.o_proj.weight"] = small({cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim});
  w[p + "self_attn.q_norm.weight"] = torch::zeros({cfg.head_dim}, torch::kFloat32);
  w[p + "self_attn.k_norm.weight"] = torch::zeros({cfg.head_dim}, torch::kFloat32);
  const int64_t linear_key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
  const int64_t linear_value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
  const int64_t linear_conv_dim = linear_key_dim * 2 + linear_value_dim;
  w[p + "linear_attn.in_proj_qkv.weight"] = small({linear_conv_dim, cfg.hidden_size});
  w[p + "linear_attn.in_proj_z.weight"] = small({linear_value_dim, cfg.hidden_size});
  w[p + "linear_attn.in_proj_b.weight"] = small({cfg.linear_num_value_heads, cfg.hidden_size});
  w[p + "linear_attn.in_proj_a.weight"] = small({cfg.linear_num_value_heads, cfg.hidden_size});
  w[p + "linear_attn.conv1d.weight"] = small({linear_conv_dim, 1, cfg.linear_conv_kernel_dim});
  w[p + "linear_attn.A_log"] = torch::zeros({cfg.linear_num_value_heads}, torch::kFloat32);
  w[p + "linear_attn.dt_bias"] = torch::zeros({cfg.linear_num_value_heads}, torch::kFloat32);
  w[p + "linear_attn.norm.weight"] = torch::ones({cfg.linear_value_head_dim}, torch::kFloat32);
  w[p + "linear_attn.out_proj.weight"] = small({cfg.hidden_size, linear_value_dim});
  w[p + "mlp.gate_proj.weight"] = torch::zeros({cfg.intermediate_size, cfg.hidden_size}, torch::kFloat32);
  w[p + "mlp.up_proj.weight"] = torch::zeros({cfg.intermediate_size, cfg.hidden_size}, torch::kFloat32);
  w[p + "mlp.down_proj.weight"] = torch::zeros({cfg.hidden_size, cfg.intermediate_size}, torch::kFloat32);
  return w;
}

void install_weights(cverl::Qwen35TextModel& model, const std::unordered_map<std::string, torch::Tensor>& weights) {
  for (const auto& item : weights) {
    model.set_weight_override(item.first, item.second);
  }
}

std::vector<torch::Tensor> remote_linear_attention_messages(const torch::Tensor& hidden_local,
                                                            const std::unordered_map<std::string, torch::Tensor>& weights,
                                                            const cverl::Qwen35TextConfig& cfg) {
  const std::string p = "model.language_model.layers.0.";
  auto x = rms_norm(hidden_local, weights.at(p + "input_layernorm.weight"), cfg.rms_norm_eps);
  const int64_t b = x.size(0);
  const int64_t s = x.size(1);
  const int64_t vh = cfg.linear_num_value_heads;
  const int64_t vd = cfg.linear_value_head_dim;
  auto mixed = dense(x, weights.at(p + "linear_attn.in_proj_qkv.weight"));
  auto z = dense(x, weights.at(p + "linear_attn.in_proj_z.weight")).reshape({b, s, vh, vd});
  auto beta = torch::sigmoid(dense(x, weights.at(p + "linear_attn.in_proj_b.weight")));
  auto a = dense(x, weights.at(p + "linear_attn.in_proj_a.weight"));
  return {
      mixed.transpose(0, 1).contiguous(),
      z.transpose(0, 1).contiguous(),
      beta.transpose(0, 1).contiguous(),
      a.transpose(0, 1).contiguous(),
  };
}

torch::Tensor linear_attention_prefix_state(const torch::Tensor& hidden,
                                            int64_t prefix_len,
                                            const std::unordered_map<std::string, torch::Tensor>& weights,
                                            const cverl::Qwen35TextConfig& cfg) {
  const std::string p = "model.language_model.layers.0.";
  auto x = rms_norm(hidden, weights.at(p + "input_layernorm.weight"), cfg.rms_norm_eps);
  const int64_t b = x.size(0);
  const int64_t seq = x.size(1);
  const int64_t kh = cfg.linear_num_key_heads;
  const int64_t vh = cfg.linear_num_value_heads;
  const int64_t kd = cfg.linear_key_head_dim;
  const int64_t vd = cfg.linear_value_head_dim;
  const int64_t key_dim = kh * kd;
  const int64_t value_dim = vh * vd;
  const int64_t conv_dim = key_dim * 2 + value_dim;

  auto mixed = dense(x, weights.at(p + "linear_attn.in_proj_qkv.weight")).transpose(1, 2);
  auto conv_opts = torch::nn::functional::Conv1dFuncOptions().padding(cfg.linear_conv_kernel_dim - 1).groups(conv_dim);
  mixed = torch::nn::functional::conv1d(mixed, weights.at(p + "linear_attn.conv1d.weight"), conv_opts);
  mixed = torch::silu(mixed.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, seq)}))
              .transpose(1, 2);
  auto qkv = mixed.split({key_dim, key_dim, value_dim}, -1);
  auto key = qkv[1].reshape({b, seq, kh, kd});
  auto value = qkv[2].reshape({b, seq, vh, vd});
  auto beta = torch::sigmoid(dense(x, weights.at(p + "linear_attn.in_proj_b.weight")));
  auto a = dense(x, weights.at(p + "linear_attn.in_proj_a.weight"));
  auto g = -torch::exp(weights.at(p + "linear_attn.A_log")) *
           torch::softplus(a + weights.at(p + "linear_attn.dt_bias"));
  if (vh / kh > 1) {
    key = key.repeat_interleave(vh / kh, 2);
  }
  key = l2norm(key).transpose(1, 2).contiguous().to(torch::kFloat32);
  value = value.transpose(1, 2).contiguous().to(torch::kFloat32);
  beta = beta.transpose(1, 2).contiguous().to(torch::kFloat32);
  g = g.transpose(1, 2).contiguous().to(torch::kFloat32);

  auto state = torch::zeros({b, vh, kd, vd}, torch::kFloat32);
  for (int64_t t = 0; t < prefix_len; ++t) {
    auto kt = key.index({torch::indexing::Slice(), torch::indexing::Slice(), t});
    auto vt = value.index({torch::indexing::Slice(), torch::indexing::Slice(), t});
    auto gt = torch::exp(g.index({torch::indexing::Slice(), torch::indexing::Slice(), t})).unsqueeze(-1).unsqueeze(-1);
    auto bt = beta.index({torch::indexing::Slice(), torch::indexing::Slice(), t}).unsqueeze(-1);
    state = state * gt;
    auto kv_mem = (state * kt.unsqueeze(-1)).sum(-2);
    auto delta = (vt - kv_mem) * bt;
    state = state + kt.unsqueeze(-1) * delta.unsqueeze(-2);
  }
  return state.contiguous();
}

void run_full_attention_case() {
  auto dir = make_model_dir("full_attention");
  cverl::HfModelLoader loader(dir.string());
  cverl::Qwen35TextModel model(std::move(loader));
  auto weights = make_weights(model.config());
  install_weights(model, weights);

  auto hidden = torch::randn({1, 4, model.config().hidden_size}, torch::kFloat32) * 0.1;
  auto dense_hidden = hidden.detach().clone().set_requires_grad(true);
  auto dense = model.forward_hidden_range_context_parallel(dense_hidden, 0, 1, cverl::distributed::ParallelGroup{}, 4, false);
  auto shard0 = cverl::distributed::context_parallel_slice_padded(hidden, 0, 2, 1, 0.0);
  auto shard1 = cverl::distributed::context_parallel_slice_padded(hidden, 1, 2, 1, 0.0);
  auto kv0 = remote_full_attention_kv(shard0, 0, weights, model.config());
  auto kv1 = remote_full_attention_kv(shard1, 2, weights, model.config());

  SendRecvCollectives rank0_collectives(
      0, 2, {torch::cat({kv1.first, kv1.second}, -1).transpose(0, 2).contiguous()});
  cverl::distributed::ParallelGroup rank0_group{8, 2, {8, 9}, &rank0_collectives};
  auto shard0_grad = shard0.detach().clone().set_requires_grad(true);
  auto out0 = model.forward_hidden_range_context_parallel(shard0_grad, 0, 1, rank0_group, 4, false);

  SendRecvCollectives rank1_collectives(
      1, 2, {torch::cat({kv0.first, kv0.second}, -1).transpose(0, 2).contiguous()});
  cverl::distributed::ParallelGroup rank1_group{9, 2, {8, 9}, &rank1_collectives};
  auto shard1_grad = shard1.detach().clone().set_requires_grad(true);
  auto out1 = model.forward_hidden_range_context_parallel(shard1_grad, 0, 1, rank1_group, 4, false);

  require(torch::allclose(out0, dense.narrow(1, 0, 2), 1.0e-5, 1.0e-5),
          "Qwen CP full-attention rank0 forward shard must match dense slice");
  require(torch::allclose(out1, dense.narrow(1, 2, 2), 1.0e-5, 1.0e-5),
          "Qwen CP full-attention rank1 forward shard must match dense slice");
  require(rank0_collectives.send_recv_calls() == 1 && rank1_collectives.send_recv_calls() == 1,
          "Qwen CP full attention should fuse K/V into one ring exchange per rank");
  auto grad0 = torch::randn_like(out0);
  auto grad1 = torch::randn_like(out1);
  auto dense_hidden0 = hidden.detach().clone().set_requires_grad(true);
  auto dense0 =
      model.forward_hidden_range_context_parallel(dense_hidden0, 0, 1, cverl::distributed::ParallelGroup{}, 4, false);
  torch::autograd::backward({dense0.narrow(1, 0, 2)}, {grad0});
  auto dense_hidden1 = hidden.detach().clone().set_requires_grad(true);
  auto dense1 =
      model.forward_hidden_range_context_parallel(dense_hidden1, 0, 1, cverl::distributed::ParallelGroup{}, 4, false);
  torch::autograd::backward({dense1.narrow(1, 2, 2)}, {grad1});
  torch::autograd::backward({out0}, {grad0});
  torch::autograd::backward({out1}, {grad1});
  require(shard0_grad.grad().defined() &&
              torch::allclose(shard0_grad.grad(), dense_hidden0.grad().narrow(1, 0, 2), 1.0e-5, 1.0e-5),
          "Qwen CP full-attention rank0 hidden gradient must match dense slice");
  require(shard1_grad.grad().defined() &&
              torch::allclose(shard1_grad.grad(), dense_hidden1.grad().narrow(1, 2, 2), 1.0e-5, 1.0e-5),
          "Qwen CP full-attention rank1 hidden gradient must match dense slice");
  std::filesystem::remove_all(dir);
}

void run_linear_attention_case() {
  auto dir = make_model_dir("linear_attention");
  cverl::HfModelLoader loader(dir.string());
  cverl::Qwen35TextModel model(std::move(loader));
  auto weights = make_weights(model.config());
  install_weights(model, weights);

  auto hidden = torch::randn({1, 4, model.config().hidden_size}, torch::kFloat32) * 0.1;
  auto dense_hidden = hidden.detach().clone().set_requires_grad(true);
  auto dense = model.forward_hidden_range_context_parallel(dense_hidden, 0, 1, cverl::distributed::ParallelGroup{}, 4, false);
  auto shard0 = cverl::distributed::context_parallel_slice_padded(hidden, 0, 2, 1, 0.0);
  auto shard1 = cverl::distributed::context_parallel_slice_padded(hidden, 1, 2, 1, 0.0);

  auto rank0_final_state_grad = torch::zeros_like(linear_attention_prefix_state(hidden, 2, weights, model.config()));
  SendRecvCollectives rank0_collectives(
      0, 2, remote_linear_attention_messages(shard1, weights, model.config()), {rank0_final_state_grad});
  cverl::distributed::ParallelGroup rank0_group{8, 2, {8, 9}, &rank0_collectives};
  auto shard0_grad = shard0.detach().clone().set_requires_grad(true);
  auto out0 = model.forward_hidden_range_context_parallel(shard0_grad, 0, 1, rank0_group, 4, false);

  auto rank1_initial_state = linear_attention_prefix_state(hidden, 2, weights, model.config());
  SendRecvCollectives rank1_collectives(
      1, 2, remote_linear_attention_messages(shard0, weights, model.config()), {rank1_initial_state});
  cverl::distributed::ParallelGroup rank1_group{9, 2, {8, 9}, &rank1_collectives};
  auto shard1_grad = shard1.detach().clone().set_requires_grad(true);
  auto out1 = model.forward_hidden_range_context_parallel(shard1_grad, 0, 1, rank1_group, 4, false);

  require(torch::allclose(out0, dense.narrow(1, 0, 2), 1.0e-5, 1.0e-5),
          "Qwen CP linear-attention rank0 forward shard must match dense slice");
  require(torch::allclose(out1, dense.narrow(1, 2, 2), 1.0e-5, 1.0e-5),
          "Qwen CP linear-attention rank1 forward shard must match dense slice");
  require(rank0_collectives.send_calls() == 1 && rank0_collectives.recv_calls() == 0,
          "Qwen CP linear-attention rank0 should send final recurrent state");
  require(rank1_collectives.send_calls() == 0 && rank1_collectives.recv_calls() == 1,
          "Qwen CP linear-attention rank1 should receive initial recurrent state");
  auto grad0 = torch::randn_like(out0);
  auto grad1 = torch::randn_like(out1);
  auto dense_hidden0 = hidden.detach().clone().set_requires_grad(true);
  auto dense0 =
      model.forward_hidden_range_context_parallel(dense_hidden0, 0, 1, cverl::distributed::ParallelGroup{}, 4, false);
  torch::autograd::backward({dense0.narrow(1, 0, 2)}, {grad0});
  auto dense_hidden1 = hidden.detach().clone().set_requires_grad(true);
  auto dense1 =
      model.forward_hidden_range_context_parallel(dense_hidden1, 0, 1, cverl::distributed::ParallelGroup{}, 4, false);
  torch::autograd::backward({dense1.narrow(1, 2, 2)}, {grad1});
  torch::autograd::backward({out0}, {grad0});
  torch::autograd::backward({out1}, {grad1});
  require(shard0_grad.grad().defined() &&
              torch::allclose(shard0_grad.grad(), dense_hidden0.grad().narrow(1, 0, 2), 1.0e-5, 1.0e-5),
          "Qwen CP linear-attention rank0 hidden gradient must match dense slice");
  require(shard1_grad.grad().defined() &&
              torch::allclose(shard1_grad.grad(), dense_hidden1.grad().narrow(1, 2, 2), 1.0e-5, 1.0e-5),
          "Qwen CP linear-attention rank1 hidden gradient must match dense slice");
  require(rank0_collectives.recv_calls() == 1,
          "Qwen CP linear-attention rank0 should receive final-state gradient during backward");
  require(rank1_collectives.send_calls() == 1,
          "Qwen CP linear-attention rank1 should send initial-state gradient during backward");
  std::filesystem::remove_all(dir);
}

}  // namespace

int main() {
  try {
    run_full_attention_case();
    run_linear_attention_case();
  } catch (const std::exception& e) {
    std::cerr << "test_qwen3_5_cp_forward failed: " << e.what() << "\n";
    return 1;
  }
  std::cout << "qwen3.5 cp forward tests passed\n";
  return 0;
}
