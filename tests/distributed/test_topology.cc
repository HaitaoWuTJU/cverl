#include "cverl/distributed/collectives.h"
#include "cverl/distributed/context_parallel.h"
#include "cverl/distributed/pipeline.h"
#include "cverl/distributed/topology.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <torch/torch.h>

namespace {

void require(bool cond, const char* msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

void require_vec_eq(const std::vector<int64_t>& actual, const std::vector<int64_t>& expected, const char* msg) {
  if (actual != expected) {
    std::cerr << msg << "\nactual:";
    for (int64_t v : actual) {
      std::cerr << " " << v;
    }
    std::cerr << "\nexpected:";
    for (int64_t v : expected) {
      std::cerr << " " << v;
    }
    std::cerr << "\n";
    throw std::runtime_error(msg);
  }
}

template <typename Fn>
void require_throws(Fn&& fn, const char* msg) {
  bool failed = false;
  try {
    fn();
  } catch (const std::exception&) {
    failed = true;
  }
  require(failed, msg);
}

class PrecomputedAllGatherCollectives final : public cverl::distributed::Collectives {
 public:
  explicit PrecomputedAllGatherCollectives(std::vector<torch::Tensor> parts) {
    world_size_ = static_cast<int64_t>(parts.size());
    responses_.push_back(std::move(parts));
  }

  explicit PrecomputedAllGatherCollectives(std::vector<std::vector<torch::Tensor>> responses)
      : responses_(std::move(responses)) {
    if (!responses_.empty()) {
      world_size_ = static_cast<int64_t>(responses_.front().size());
    }
  }

  PrecomputedAllGatherCollectives(std::vector<std::vector<torch::Tensor>> responses,
                                  std::vector<torch::Tensor> send_recv_responses,
                                  int64_t world_size = 2)
      : responses_(std::move(responses)), send_recv_responses_(std::move(send_recv_responses)), world_size_(world_size) {
    if (!responses_.empty()) {
      world_size_ = static_cast<int64_t>(responses_.front().size());
    }
  }

  int64_t rank() const override { return 0; }
  int64_t world_size() const override {
    return world_size_;
  }
  void barrier() override {}

  torch::Tensor broadcast(const torch::Tensor& input,
                          int64_t /*root*/,
                          const std::vector<int64_t>& /*group*/) override {
    return input;
  }

  torch::Tensor all_reduce(const torch::Tensor& input,
                           cverl::distributed::ReduceOp /*op*/,
                           const std::vector<int64_t>& /*group*/) override {
    return input;
  }

  torch::Tensor all_gather(const torch::Tensor& /*input*/,
                           const std::vector<int64_t>& /*group*/,
                           int64_t dim) override {
    const size_t index = std::min(call_count_, responses_.size() - 1);
    ++call_count_;
    return torch::cat(responses_.at(index), dim);
  }

  torch::Tensor reduce_scatter(const torch::Tensor& input,
                               cverl::distributed::ReduceOp /*op*/,
                               const std::vector<int64_t>& /*group*/,
                               int64_t dim) override {
    if (dim != 0 || responses_.empty()) {
      if (dim != 0 || world_size_ <= 1) {
        return input;
      }
      return input.narrow(0, 0, input.size(0) / world_size_).contiguous();
    }
    return input.narrow(0, 0, input.size(0) / world_size_).contiguous();
  }

  void send(const torch::Tensor& /*input*/, int64_t /*peer*/) override {}
  torch::Tensor recv_like(const torch::Tensor& like, int64_t /*peer*/) override { return torch::empty_like(like); }
  torch::Tensor send_recv(const torch::Tensor& /*input*/,
                          int64_t /*send_peer*/,
                          const torch::Tensor& like,
                          int64_t /*recv_peer*/) override {
    if (send_recv_responses_.empty()) {
      return torch::empty_like(like);
    }
    const size_t index = std::min(send_recv_call_count_, send_recv_responses_.size() - 1);
    ++send_recv_call_count_;
    return send_recv_responses_.at(index);
  }

 private:
  std::vector<std::vector<torch::Tensor>> responses_;
  std::vector<torch::Tensor> send_recv_responses_;
  int64_t world_size_ = 0;
  size_t call_count_ = 0;
  size_t send_recv_call_count_ = 0;
};

cverl::distributed::ClusterSpec make_spec() {
  cverl::distributed::ClusterSpec spec;
  spec.parallel.data_parallel = 2;
  spec.parallel.pipeline_parallel = 2;
  spec.parallel.context_parallel = 2;
  spec.parallel.tensor_parallel = 4;
  spec.parallel.micro_batches = 4;
  spec.world_size = 32;
  spec.rank = 13;
  spec.local_world_size = 8;
  spec.local_rank = 5;
  spec.gpus_per_node = 8;
  spec.network.socket_ifnames = {"eth0", "ib0"};
  spec.network.ib_hcas = {"mlx5_0", "mlx5_1"};
  spec.network.nccl_ib_gid_index = 3;
  return spec;
}

void test_rank_mapping() {
  cverl::distributed::Topology topology(make_spec());
  auto info = topology.local_rank_info();
  require(info.data_rank == 0, "data rank");
  require(info.pipeline_rank == 1, "pipeline rank");
  require(info.context_rank == 1, "context rank");
  require(info.tensor_rank == 1, "tensor rank");
  require_vec_eq(info.tensor_group, {12, 13, 14, 15}, "tensor group");
  require_vec_eq(info.pipeline_group, {5, 13}, "pipeline group");
  require_vec_eq(info.context_group, {9, 13}, "context group");
  require_vec_eq(info.data_group, {13, 29}, "data group");
  require_vec_eq(info.model_group,
                 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
                 "model group");
  require(topology.global_rank(1, 1, 1, 1) == 29, "global rank");
  require(topology.global_rank(1, 1, 1) == 25, "legacy global rank uses context rank 0");
}

void test_nccl_env() {
  cverl::distributed::Topology topology(make_spec());
  auto env = topology.nccl_env();
  require(env.at("NCCL_SOCKET_IFNAME") == "eth0,ib0", "socket ifnames");
  require(env.at("NCCL_IB_HCA") == "mlx5_0,mlx5_1", "ib hcas");
  require(env.at("NCCL_IB_GID_INDEX") == "3", "gid index");
  require(env.at("NCCL_NVLS_ENABLE") == "1", "nvls");
  require(env.at("NCCL_ASYNC_ERROR_HANDLING") == "1", "async error handling");

  auto spec = make_spec();
  spec.network.enable_ib = false;
  spec.network.enable_p2p = false;
  spec.network.enable_nvls = false;
  spec.network.nccl_topo_file = "/tmp/topo.xml";
  cverl::distributed::Topology disabled(spec);
  env = disabled.nccl_env();
  require(env.at("NCCL_IB_DISABLE") == "1", "ib disabled");
  require(env.at("NCCL_P2P_DISABLE") == "1", "p2p disabled");
  require(env.at("NCCL_NVLS_ENABLE") == "0", "nvls disabled");
  require(env.at("NCCL_TOPO_FILE") == "/tmp/topo.xml", "topo file");
}

void test_pipeline_layer_ranges() {
  cverl::distributed::Topology topology(make_spec());
  auto r0 = topology.pipeline_layer_range(24, 0);
  auto r1 = topology.pipeline_layer_range(24, 1);
  require(r0.begin == 0 && r0.end == 12, "pipeline range 0");
  require(r1.begin == 12 && r1.end == 24, "pipeline range 1");

  auto uneven0 = topology.pipeline_layer_range(25, 0);
  auto uneven1 = topology.pipeline_layer_range(25, 1);
  require(uneven0.begin == 0 && uneven0.end == 13, "pipeline uneven range 0");
  require(uneven1.begin == 13 && uneven1.end == 25, "pipeline uneven range 1");

  auto local = topology.local_pipeline_layer_range(25);
  require(local.begin == 13 && local.end == 25, "local pipeline range");

  auto empty = topology.pipeline_layer_range(1, 1);
  require(empty.begin == 1 && empty.end == 1, "empty pipeline range");
}

void test_invalid_configs() {
  auto spec = make_spec();
  spec.world_size = 15;
  bool failed = false;
  try {
    cverl::distributed::Topology topology(spec);
  } catch (const std::invalid_argument&) {
    failed = true;
  }
  require(failed, "invalid world size rejected");

  spec = make_spec();
  spec.parallel.tensor_parallel = 16;
  spec.parallel.data_parallel = 1;
  spec.parallel.pipeline_parallel = 1;
  spec.parallel.context_parallel = 1;
  spec.world_size = 16;
  failed = false;
  try {
    cverl::distributed::Topology topology(spec);
  } catch (const std::invalid_argument&) {
    failed = true;
  }
  require(failed, "oversized tensor parallel rejected");

  spec = make_spec();
  spec.parallel.micro_batches = 1;
  failed = false;
  try {
    cverl::distributed::Topology topology(spec);
  } catch (const std::invalid_argument&) {
    failed = true;
  }
  require(failed, "too few micro batches rejected");

  cverl::distributed::Topology topology(make_spec());
  require_throws([&]() { (void)topology.global_rank(2, 0, 0); }, "global rank out-of-range rejected");
  require_throws([&]() { (void)topology.global_rank(0, 0, 2, 0); }, "context rank out-of-range rejected");
  require_throws([&]() { (void)topology.rank_info(32); }, "rank_info out-of-range rejected");
  require_throws([&]() { (void)topology.pipeline_layer_range(1, 2); }, "pipeline rank out-of-range rejected");
  require_throws([&]() { (void)topology.pipeline_layer_range(-1, 0); }, "negative layer count rejected");
}

void test_single_process_collectives() {
  cverl::distributed::SingleProcessCollectives collectives;
  auto x = torch::tensor({1.0f, 2.0f, 3.0f});
  auto y = collectives.all_reduce(x, cverl::distributed::ReduceOp::Sum, {0});
  require(torch::allclose(x, y), "single process all_reduce");
  auto z = collectives.all_gather(x, {0}, 0);
  require(torch::allclose(x, z), "single process all_gather");
  auto s = collectives.reduce_scatter(x, cverl::distributed::ReduceOp::Max, {0}, 0);
  require(torch::allclose(x, s), "single process reduce_scatter");
}

void test_cluster_spec_from_env() {
  setenv("WORLD_SIZE", "4", 1);
  setenv("RANK", "3", 1);
  setenv("LOCAL_WORLD_SIZE", "2", 1);
  setenv("LOCAL_RANK", "1", 1);
  setenv("NODE_RANK", "9", 1);
  setenv("CVERL_GPUS_PER_NODE", "4", 1);

  cverl::distributed::ParallelDims dims;
  dims.data_parallel = 2;
  dims.tensor_parallel = 2;
  dims.pipeline_parallel = 1;
  dims.context_parallel = 1;
  dims.micro_batches = 1;
  auto spec = cverl::distributed::cluster_spec_from_env(dims);
  require(spec.world_size == 4, "env world size");
  require(spec.rank == 3, "env rank");
  require(spec.local_world_size == 2, "env local world size");
  require(spec.local_rank == 1, "env local rank");
  require(spec.node_rank == 9, "env node rank");
  require(spec.gpus_per_node == 4, "env gpus per node");
  cverl::distributed::Topology topology(spec);
  auto info = topology.local_rank_info();
  require_vec_eq(info.tensor_group, {2, 3}, "env tensor group");
  require_vec_eq(info.data_group, {1, 3}, "env data group");
}

void test_pipeline_peers() {
  cverl::distributed::Topology topology(make_spec());
  auto info = topology.local_rank_info();
  auto peers = cverl::distributed::pipeline_peers(topology, info);
  require(!peers.is_first_stage, "rank 13 is not first PP stage");
  require(peers.is_last_stage, "rank 13 is last PP stage");
  require(peers.previous_rank == 5, "previous PP peer");
  require(peers.next_rank == -1, "no next PP peer");
  require(cverl::distributed::pipeline_warmup_micro_batches(topology, info) == 0, "last stage warmup");

  auto first = topology.rank_info(5);
  peers = cverl::distributed::pipeline_peers(topology, first);
  require(peers.is_first_stage, "rank 5 is first PP stage");
  require(!peers.is_last_stage, "rank 5 is not last PP stage");
  require(peers.next_rank == 13, "next PP peer");
  require(cverl::distributed::pipeline_warmup_micro_batches(topology, first) == 1, "first stage warmup");
}

void test_pipeline_1f1b_schedule() {
  cverl::distributed::ClusterSpec spec;
  spec.parallel.data_parallel = 1;
  spec.parallel.pipeline_parallel = 4;
  spec.parallel.context_parallel = 1;
  spec.parallel.tensor_parallel = 1;
  spec.parallel.micro_batches = 8;
  spec.world_size = 4;
  spec.local_world_size = 4;
  spec.gpus_per_node = 4;

  {
    spec.rank = 0;
    cverl::distributed::Topology topology(spec);
    auto actions = cverl::distributed::pipeline_1f1b_schedule(topology, topology.local_rank_info());
    require(actions.size() == 16, "stage0 action count");
    require(actions[0].phase == cverl::distributed::PipelineSchedulePhase::Warmup &&
                actions[0].op == cverl::distributed::PipelineScheduleOp::Forward &&
                actions[0].micro_batch == 0 && !actions[0].recv_forward && actions[0].send_forward,
            "stage0 warmup fwd0");
    require(actions[2].micro_batch == 2 && actions[2].op == cverl::distributed::PipelineScheduleOp::Forward,
            "stage0 warmup fwd2");
    require(actions[3].phase == cverl::distributed::PipelineSchedulePhase::Steady &&
                actions[3].op == cverl::distributed::PipelineScheduleOp::Forward &&
                actions[3].micro_batch == 3,
            "stage0 steady fwd3");
    require(actions[4].op == cverl::distributed::PipelineScheduleOp::Backward &&
                actions[4].micro_batch == 0 && actions[4].recv_backward && !actions[4].send_backward,
            "stage0 steady bwd0");
    require(actions.back().phase == cverl::distributed::PipelineSchedulePhase::Cooldown &&
                actions.back().op == cverl::distributed::PipelineScheduleOp::Backward &&
                actions.back().micro_batch == 7,
            "stage0 cooldown bwd7");
    require(cverl::distributed::pipeline_schedule_max_live_activations(actions) == 4, "stage0 max live");
  }

  {
    spec.rank = 3;
    cverl::distributed::Topology topology(spec);
    auto actions = cverl::distributed::pipeline_1f1b_schedule(topology, topology.local_rank_info());
    require(actions.size() == 16, "last stage action count");
    require(actions[0].phase == cverl::distributed::PipelineSchedulePhase::Steady &&
                actions[0].op == cverl::distributed::PipelineScheduleOp::Forward &&
                actions[0].micro_batch == 0 && actions[0].recv_forward && !actions[0].send_forward,
            "last stage steady fwd0");
    require(actions[1].op == cverl::distributed::PipelineScheduleOp::Backward &&
                actions[1].micro_batch == 0 && !actions[1].recv_backward && actions[1].send_backward,
            "last stage steady bwd0");
    require(cverl::distributed::pipeline_schedule_max_live_activations(actions) == 1, "last stage max live");
  }
}

void test_context_parallel_slice() {
  auto x = torch::arange(2 * 8 * 3, torch::kFloat32).view({2, 8, 3});
  auto left = cverl::distributed::context_parallel_slice(x, 0, 2, 1);
  auto right = cverl::distributed::context_parallel_slice(x, 1, 2, 1);
  require(torch::allclose(left, x.index({torch::indexing::Slice(), torch::indexing::Slice(0, 4)})),
          "context slice left");
  require(torch::allclose(right, x.index({torch::indexing::Slice(), torch::indexing::Slice(4, 8)})),
          "context slice right");
  require_throws([&]() { (void)cverl::distributed::context_parallel_slice(x, 0, 3, 1); },
                 "non-divisible context slice rejected");
}

void test_context_parallel_padded_slice_and_gather() {
  auto x = torch::arange(5, torch::kFloat32).view({1, 5});
  require(cverl::distributed::context_parallel_padded_length(5, 2) == 6, "context padded length");
  auto r0 = cverl::distributed::context_parallel_sequence_range(5, 0, 2);
  auto r1 = cverl::distributed::context_parallel_sequence_range(5, 1, 2);
  require(r0.begin == 0 && r0.end == 3 && r0.padded_end == 3, "context range 0");
  require(r1.begin == 3 && r1.end == 5 && r1.padded_end == 6, "context range 1");

  auto s0 = cverl::distributed::context_parallel_slice_padded(x, 0, 2, 1, -1.0);
  auto s1 = cverl::distributed::context_parallel_slice_padded(x, 1, 2, 1, -1.0);
  require(torch::allclose(s0, torch::tensor({0.0f, 1.0f, 2.0f}).view({1, 3})), "padded context slice 0");
  require(torch::allclose(s1, torch::tensor({3.0f, 4.0f, -1.0f}).view({1, 3})), "padded context slice 1");

  PrecomputedAllGatherCollectives collectives({s0.transpose(0, 1).contiguous(), s1.transpose(0, 1).contiguous()});
  auto gathered = cverl::distributed::context_parallel_gather_padded(s0, collectives, {0, 1}, 1, 5);
  require(torch::allclose(gathered, x), "padded context gather trims padding");
}

void test_context_parallel_ring_schedule() {
  auto schedule = cverl::distributed::context_parallel_ring_schedule({0, 1, 2, 3}, 2);
  require(schedule.size() == 4, "context ring schedule size");
  require(schedule[0].kv_rank == 2 && schedule[0].send_rank == 3 && schedule[0].recv_rank == 1,
          "context ring step 0");
  require(schedule[1].kv_rank == 1 && schedule[2].kv_rank == 0 && schedule[3].kv_rank == 3,
          "context ring kv order");
  require(cverl::distributed::context_parallel_group_index({8, 9}, 9) == 1, "context group index");
  require_throws([&]() { (void)cverl::distributed::context_parallel_ring_schedule({0, 1}, 3); },
                 "context ring rejects non-member rank");
}

void test_context_parallel_causal_attention() {
  torch::manual_seed(11);
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  auto q = torch::randn({1, 2, 6, 4}, opts);
  auto k = torch::randn({1, 2, 6, 4}, opts);
  auto v = torch::randn({1, 2, 6, 3}, opts);
  const double scale = 1.0 / std::sqrt(4.0);

  auto dense_scores = torch::matmul(q, k.transpose(2, 3)) * scale;
  auto mask = torch::ones({6, 6}, torch::TensorOptions().dtype(torch::kBool)).triu(1);
  dense_scores = dense_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1.0e9);
  auto dense = torch::matmul(torch::softmax(dense_scores, -1), v);

  auto local_q = q.narrow(2, 3, 3).contiguous();
  auto local = cverl::distributed::context_parallel_causal_attention(local_q, k, v, 3, scale);
  require(torch::allclose(local, dense.narrow(2, 3, 3), 1.0e-5, 1.0e-5), "CP local causal attention");

  auto stream_q = q.narrow(2, 3, 3).contiguous().set_requires_grad(true);
  auto stream_k0 = k.narrow(2, 0, 3).contiguous().set_requires_grad(true);
  auto stream_v0 = v.narrow(2, 0, 3).contiguous().set_requires_grad(true);
  auto stream_k1 = k.narrow(2, 3, 3).contiguous().set_requires_grad(true);
  auto stream_v1 = v.narrow(2, 3, 3).contiguous().set_requires_grad(true);
  auto stream_out = cverl::distributed::context_parallel_causal_attention_streaming_blocks(
      stream_q, {stream_k1, stream_k0}, {stream_v1, stream_v0}, {3, 0}, 3, 6, scale);
  require(torch::allclose(stream_out, dense.narrow(2, 3, 3), 1.0e-5, 1.0e-5),
          "CP streaming causal attention matches dense regardless of ring block order");
  auto future_first = cverl::distributed::context_parallel_causal_attention_streaming_blocks(
      q.narrow(2, 0, 3).contiguous(), {k.narrow(2, 3, 3).contiguous(), k.narrow(2, 0, 3).contiguous()},
      {v.narrow(2, 3, 3).contiguous(), v.narrow(2, 0, 3).contiguous()}, {3, 0}, 0, 6, scale);
  require(torch::allclose(future_first, dense.narrow(2, 0, 3), 1.0e-5, 1.0e-5),
          "CP streaming ignores all-future ring blocks until valid KV arrives");
  auto stream_dense_q = q.detach().clone().set_requires_grad(true);
  auto stream_dense_k = k.detach().clone().set_requires_grad(true);
  auto stream_dense_v = v.detach().clone().set_requires_grad(true);
  auto stream_dense_local =
      cverl::distributed::context_parallel_causal_attention(stream_dense_q.narrow(2, 3, 3).contiguous(),
                                                            stream_dense_k,
                                                            stream_dense_v,
                                                            3,
                                                            scale);
  stream_dense_local.sum().backward();
  stream_out.sum().backward();
  require(stream_q.grad().defined() &&
              torch::allclose(stream_q.grad(), stream_dense_q.grad().narrow(2, 3, 3), 1.0e-5, 1.0e-5),
          "CP streaming query gradient matches dense");
  require(stream_k0.grad().defined() &&
              torch::allclose(stream_k0.grad(), stream_dense_k.grad().narrow(2, 0, 3), 1.0e-5, 1.0e-5),
          "CP streaming key block 0 gradient matches dense");
  require(stream_k1.grad().defined() &&
              torch::allclose(stream_k1.grad(), stream_dense_k.grad().narrow(2, 3, 3), 1.0e-5, 1.0e-5),
          "CP streaming key block 1 gradient matches dense");
  require(stream_v0.grad().defined() &&
              torch::allclose(stream_v0.grad(), stream_dense_v.grad().narrow(2, 0, 3), 1.0e-5, 1.0e-5),
          "CP streaming value block 0 gradient matches dense");
  require(stream_v1.grad().defined() &&
              torch::allclose(stream_v1.grad(), stream_dense_v.grad().narrow(2, 3, 3), 1.0e-5, 1.0e-5),
          "CP streaming value block 1 gradient matches dense");

  auto recompute_q = q.narrow(2, 0, 3).contiguous().set_requires_grad(true);
  auto recompute_k0 = k.narrow(2, 0, 3).contiguous();
  auto recompute_k1 = k.narrow(2, 3, 3).contiguous();
  auto recompute_v0 = v.narrow(2, 0, 3).contiguous();
  auto recompute_v1 = v.narrow(2, 3, 3).contiguous();
  auto recompute_k_ring = torch::cat({recompute_k0, recompute_k1}, 2).set_requires_grad(true);
  auto recompute_v_ring = torch::cat({recompute_v0, recompute_v1}, 2).set_requires_grad(true);
  auto recompute_dense_q = q.detach().clone().set_requires_grad(true);
  auto recompute_dense_k = k.detach().clone().set_requires_grad(true);
  auto recompute_dense_v = v.detach().clone().set_requires_grad(true);
  auto recompute_dense_local =
      cverl::distributed::context_parallel_causal_attention(recompute_dense_q.narrow(2, 0, 3).contiguous(),
                                                            recompute_dense_k,
                                                            recompute_dense_v,
                                                            0,
                                                            scale);
  recompute_dense_local.sum().backward();
  auto recompute_out = cverl::distributed::context_parallel_causal_attention_ring_blocks_recompute(
      recompute_q, recompute_k_ring, recompute_v_ring, {0, 3}, 0, 6, 3, scale);
  require(torch::allclose(recompute_out, dense.narrow(2, 0, 3), 1.0e-5, 1.0e-5),
          "CP recompute ring attention forward matches dense");
  recompute_out.sum().backward();
  require(recompute_q.grad().defined() &&
              torch::allclose(recompute_q.grad(), recompute_dense_q.grad().narrow(2, 0, 3), 1.0e-5, 1.0e-5),
          "CP recompute ring attention query gradient matches dense");
  require(recompute_k_ring.grad().defined() &&
              torch::allclose(
                  recompute_k_ring.grad().narrow(2, 0, 3), recompute_dense_k.grad().narrow(2, 0, 3), 1.0e-5, 1.0e-5),
          "CP recompute ring attention key block 0 gradient matches dense");
  require(torch::allclose(
              recompute_k_ring.grad().narrow(2, 3, 3), recompute_dense_k.grad().narrow(2, 3, 3), 1.0e-5, 1.0e-5),
          "CP recompute ring attention key block 1 gradient matches dense");
  require(recompute_v_ring.grad().defined() &&
              torch::allclose(
                  recompute_v_ring.grad().narrow(2, 0, 3), recompute_dense_v.grad().narrow(2, 0, 3), 1.0e-5, 1.0e-5),
          "CP recompute ring attention value block 0 gradient matches dense");
  require(torch::allclose(
              recompute_v_ring.grad().narrow(2, 3, 3), recompute_dense_v.grad().narrow(2, 3, 3), 1.0e-5, 1.0e-5),
          "CP recompute ring attention value block 1 gradient matches dense");

  auto ring_x0 = k.narrow(2, 0, 3).contiguous().set_requires_grad(true);
  auto ring_zero = torch::zeros_like(k.narrow(2, 0, 3).transpose(0, 2).contiguous());
  PrecomputedAllGatherCollectives ring_exchange_collectives(
      {}, {k.narrow(2, 3, 3).transpose(0, 2).contiguous(), ring_zero, ring_zero});
  auto ring_exchanged =
      cverl::distributed::context_parallel_ring_exchange_autograd(ring_x0, ring_exchange_collectives, {0, 1}, 0, 2);
  require(torch::allclose(ring_exchanged, k, 1.0e-5, 1.0e-5), "CP ring exchange forward order");
  ring_exchanged.sum().backward();
  require(ring_x0.grad().defined() && torch::allclose(ring_x0.grad(), torch::ones_like(ring_x0), 1.0e-5, 1.0e-5),
          "CP ring exchange backward returns owner gradient");

  auto dense_q = q.detach().clone().set_requires_grad(true);
  auto dense_k = k.detach().clone().set_requires_grad(true);
  auto dense_v = v.detach().clone().set_requires_grad(true);
  auto dense_local = cverl::distributed::context_parallel_causal_attention(dense_q.narrow(2, 0, 3).contiguous(),
                                                                           dense_k,
                                                                           dense_v,
                                                                           0,
                                                                           scale);
  dense_local.sum().backward();

  auto q0 = q.narrow(2, 0, 3).contiguous().set_requires_grad(true);
  auto k0 = k.narrow(2, 0, 3).contiguous().set_requires_grad(true);
  auto v0 = v.narrow(2, 0, 3).contiguous().set_requires_grad(true);
  auto k1 = k.narrow(2, 3, 3).contiguous();
  auto v1 = v.narrow(2, 3, 3).contiguous();
  PrecomputedAllGatherCollectives collectives(std::vector<std::vector<torch::Tensor>>{
      {k0.transpose(0, 2).contiguous(), k1.transpose(0, 2).contiguous()},
      {v0.transpose(0, 2).contiguous(), v1.transpose(0, 2).contiguous()}});
  auto gathered =
      cverl::distributed::context_parallel_causal_attention_gather_kv(q0, k0, v0, collectives, {0, 1}, 0, 6, scale);
  require(torch::allclose(gathered, dense.narrow(2, 0, 3), 1.0e-5, 1.0e-5), "CP gathered KV causal attention");
  PrecomputedAllGatherCollectives ring_collectives(std::vector<std::vector<torch::Tensor>>{
      {k0.transpose(0, 2).contiguous(), k1.transpose(0, 2).contiguous()},
      {v0.transpose(0, 2).contiguous(), v1.transpose(0, 2).contiguous()}});
  auto ring_gathered = cverl::distributed::context_parallel_causal_attention_ring_gather_kv(
      q0, k0, v0, ring_collectives, {0, 1}, 0, 6, scale);
  require(torch::allclose(ring_gathered, dense.narrow(2, 0, 3), 1.0e-5, 1.0e-5),
          "CP ring-order gathered KV causal attention");
  auto q0_exchange = q.narrow(2, 0, 3).contiguous().set_requires_grad(true);
  auto k0_exchange = k.narrow(2, 0, 3).contiguous().set_requires_grad(true);
  auto v0_exchange = v.narrow(2, 0, 3).contiguous().set_requires_grad(true);
  PrecomputedAllGatherCollectives attention_exchange_collectives(
      {},
      {k1.transpose(0, 2).contiguous(),
       v1.transpose(0, 2).contiguous(),
       torch::zeros_like(v0_exchange.transpose(0, 2).contiguous()),
       torch::zeros_like(v0_exchange.transpose(0, 2).contiguous()),
       torch::zeros_like(k0_exchange.transpose(0, 2).contiguous()),
       torch::zeros_like(k0_exchange.transpose(0, 2).contiguous())});
  auto ring_exchanged_attention = cverl::distributed::context_parallel_causal_attention_ring_exchange_kv(
      q0_exchange, k0_exchange, v0_exchange, attention_exchange_collectives, {0, 1}, 0, 6, scale);
  require(torch::allclose(ring_exchanged_attention, dense.narrow(2, 0, 3), 1.0e-5, 1.0e-5),
          "CP ring-exchange KV causal attention");
  ring_exchanged_attention.sum().backward();
  require(q0_exchange.grad().defined() &&
              torch::allclose(q0_exchange.grad(), dense_q.grad().narrow(2, 0, 3), 1.0e-5, 1.0e-5),
          "CP ring-exchange query gradient matches dense");
  require(k0_exchange.grad().defined() &&
              torch::allclose(k0_exchange.grad(), dense_k.grad().narrow(2, 0, 3), 1.0e-5, 1.0e-5),
          "CP ring-exchange key gradient matches dense");
  require(v0_exchange.grad().defined() &&
              torch::allclose(v0_exchange.grad(), dense_v.grad().narrow(2, 0, 3), 1.0e-5, 1.0e-5),
          "CP ring-exchange value gradient matches dense");
  gathered.sum().backward();
  require(q0.grad().defined() && torch::allclose(q0.grad(), dense_q.grad().narrow(2, 0, 3), 1.0e-5, 1.0e-5),
          "CP gathered KV query gradient matches dense local-output gradient");
  require(k0.grad().defined() && torch::allclose(k0.grad(), dense_k.grad().narrow(2, 0, 3), 1.0e-5, 1.0e-5),
          "CP gathered KV key gradient matches dense local-output gradient");
  require(v0.grad().defined() && torch::allclose(v0.grad(), dense_v.grad().narrow(2, 0, 3), 1.0e-5, 1.0e-5),
          "CP gathered KV value gradient matches dense local-output gradient");

  auto q1 = q.narrow(2, 3, 3).contiguous();
  PrecomputedAllGatherCollectives padded_collectives(std::vector<std::vector<torch::Tensor>>{
      {k0.detach().transpose(0, 2).contiguous(), k1.transpose(0, 2).contiguous()},
      {v0.detach().transpose(0, 2).contiguous(), v1.transpose(0, 2).contiguous()}});
  auto padded =
      cverl::distributed::context_parallel_causal_attention_gather_kv(q1, k1, v1, padded_collectives, {0, 1}, 1, 5, scale);
  auto dense5 = cverl::distributed::context_parallel_causal_attention(
      q.narrow(2, 3, 2).contiguous(), k.narrow(2, 0, 5).contiguous(), v.narrow(2, 0, 5).contiguous(), 3, scale);
  require(torch::allclose(padded.narrow(2, 0, 2), dense5, 1.0e-5, 1.0e-5),
          "CP padded tail valid queries match dense attention");
  require(padded.narrow(2, 2, 1).abs().sum().item<float>() == 0.0f, "CP padded tail query output is zero");

  require_throws([&]() { (void)cverl::distributed::context_parallel_causal_attention(local_q, k, v, 4, scale); },
                 "CP causal attention rejects out-of-range query shard");
}

void test_context_parallel_ring_exchange_owner_gradient_cp3() {
  torch::manual_seed(17);
  auto local = torch::arange(100.0, 112.0, torch::kFloat32).reshape({2, 2, 3}).set_requires_grad(true);
  auto owner0 = torch::arange(0.0, 12.0, torch::kFloat32).reshape({2, 2, 3});
  auto owner2 = torch::arange(200.0, 212.0, torch::kFloat32).reshape({2, 2, 3});

  auto moved_owner0 = owner0.transpose(0, 1).contiguous();
  auto moved_owner2 = owner2.transpose(0, 1).contiguous();
  auto remote_owner1_from_rank0 = torch::full_like(local.transpose(0, 1).contiguous(), 3.0);
  auto remote_owner1_from_rank2 = torch::full_like(local.transpose(0, 1).contiguous(), 5.0);
  auto zeros = torch::zeros_like(local.transpose(0, 1).contiguous());

  PrecomputedAllGatherCollectives collectives(
      {},
      {
          moved_owner0,
          moved_owner2,
          zeros,
          zeros,
          remote_owner1_from_rank0,
          remote_owner1_from_rank2,
          zeros,
          zeros,
      },
      3);

  auto exchanged =
      cverl::distributed::context_parallel_ring_exchange_autograd(local, collectives, {0, 1, 2}, 1, 1);
  auto expected = torch::cat({local.detach(), owner0, owner2}, 1);
  require(torch::allclose(exchanged, expected, 1.0e-5, 1.0e-5),
          "CP3 ring exchange schedule-order forward on nonzero sequence dim");

  auto grad_owner0 = torch::full_like(owner0, 11.0);
  auto grad_owner1 = torch::full_like(local, 7.0);
  auto grad_owner2 = torch::full_like(owner2, 13.0);
  torch::autograd::backward({exchanged}, {torch::cat({grad_owner1, grad_owner0, grad_owner2}, 1)});

  auto expected_local_grad =
      grad_owner1 + remote_owner1_from_rank0.transpose(0, 1) + remote_owner1_from_rank2.transpose(0, 1);
  require(local.grad().defined() && torch::allclose(local.grad(), expected_local_grad, 1.0e-5, 1.0e-5),
          "CP3 ring exchange backward accumulates remote owner gradients");
}

void test_dtype_names() {
  require(cverl::distributed::dtype_policy_name(cverl::distributed::DTypePolicy::Float32) == "float32", "float32 name");
  require(cverl::distributed::dtype_policy_name(cverl::distributed::DTypePolicy::Float16) == "float16", "float16 name");
  require(cverl::distributed::dtype_policy_name(cverl::distributed::DTypePolicy::BFloat16) == "bfloat16", "bfloat16 name");
}

}  // namespace

int main() {
  try {
    test_rank_mapping();
    test_nccl_env();
    test_pipeline_layer_ranges();
    test_invalid_configs();
    test_single_process_collectives();
    test_cluster_spec_from_env();
    test_pipeline_peers();
    test_pipeline_1f1b_schedule();
    test_context_parallel_slice();
    test_context_parallel_padded_slice_and_gather();
    test_context_parallel_ring_schedule();
    test_context_parallel_causal_attention();
    test_context_parallel_ring_exchange_owner_gradient_cp3();
    test_dtype_names();
  } catch (const std::exception& e) {
    std::cerr << "test_topology failed: " << e.what() << "\n";
    return 1;
  }
  std::cout << "distributed topology tests passed\n";
  return 0;
}
