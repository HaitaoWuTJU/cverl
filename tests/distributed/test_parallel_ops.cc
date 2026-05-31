#include "cverl/distributed/parallel_ops.h"

#include <iostream>
#include <stdexcept>
#include <vector>

#include <torch/torch.h>

namespace {

class CountingCollectives final : public cverl::distributed::Collectives {
 public:
  explicit CountingCollectives(int64_t world_size = 1, int64_t rank = 0) : world_size_(world_size), rank_(rank) {}

  int64_t all_reduce_calls = 0;
  int64_t reduce_scatter_calls = 0;
  int64_t last_all_reduce_numel = 0;
  int64_t last_reduce_scatter_numel = 0;
  torch::ScalarType last_all_reduce_dtype = torch::kFloat32;
  torch::ScalarType last_reduce_scatter_dtype = torch::kFloat32;

  int64_t rank() const override { return rank_; }
  int64_t world_size() const override { return world_size_; }
  void barrier() override {}

  torch::Tensor broadcast(const torch::Tensor& input,
                          int64_t root,
                          const std::vector<int64_t>& group) override {
    require_group(group);
    if (root != 0) {
      throw std::invalid_argument("CountingCollectives root must be 0");
    }
    return input.clone();
  }

  torch::Tensor all_reduce(const torch::Tensor& input,
                           cverl::distributed::ReduceOp /*op*/,
                           const std::vector<int64_t>& group) override {
    require_group(group);
    ++all_reduce_calls;
    last_all_reduce_numel = input.numel();
    last_all_reduce_dtype = input.scalar_type();
    return input.clone();
  }

  torch::Tensor all_gather(const torch::Tensor& input,
                           const std::vector<int64_t>& group,
                           int64_t /*dim*/) override {
    require_group(group);
    return input.clone();
  }

  torch::Tensor reduce_scatter(const torch::Tensor& input,
                               cverl::distributed::ReduceOp /*op*/,
                               const std::vector<int64_t>& group,
                               int64_t /*dim*/) override {
    require_group(group);
    ++reduce_scatter_calls;
    last_reduce_scatter_numel = input.numel();
    last_reduce_scatter_dtype = input.scalar_type();
    if (input.size(0) % world_size_ != 0) {
      throw std::invalid_argument("CountingCollectives reduce_scatter expects an even first dimension");
    }
    const int64_t shard = input.size(0) / world_size_;
    return input.narrow(0, rank_ * shard, shard).clone();
  }

  void send(const torch::Tensor& /*input*/, int64_t peer) override {
    if (peer != 0) {
      throw std::invalid_argument("CountingCollectives peer must be 0");
    }
  }

  torch::Tensor recv_like(const torch::Tensor& like, int64_t peer) override {
    if (peer != 0) {
      throw std::invalid_argument("CountingCollectives peer must be 0");
    }
    return torch::empty_like(like);
  }

 private:
  void require_group(const std::vector<int64_t>& group) const {
    if (static_cast<int64_t>(group.size()) != world_size_) {
      throw std::invalid_argument("CountingCollectives group size mismatch");
    }
    for (int64_t i = 0; i < world_size_; ++i) {
      if (group[static_cast<size_t>(i)] != i) {
        throw std::invalid_argument("CountingCollectives only supports ordered full-world groups");
      }
    }
  }

  int64_t world_size_ = 1;
  int64_t rank_ = 0;
};

void require_allclose(const torch::Tensor& actual,
                      const torch::Tensor& expected,
                      const char* msg,
                      double rtol = 1.0e-5,
                      double atol = 1.0e-6) {
  if (!torch::allclose(actual, expected, rtol, atol)) {
    std::cerr << msg << "\nactual=" << actual << "\nexpected=" << expected << "\n";
    throw std::runtime_error(msg);
  }
}

template <typename Fn>
void require_throws(Fn&& fn, const char* msg) {
  bool failed = false;
  try {
    fn();
  } catch (const std::invalid_argument&) {
    failed = true;
  }
  if (!failed) {
    throw std::runtime_error(msg);
  }
}

torch::Tensor dense_linear(const torch::Tensor& input, const torch::Tensor& weight) {
  return torch::matmul(input, weight.transpose(0, 1));
}

void test_shard_dim() {
  auto x = torch::arange(24, torch::kFloat32).reshape({2, 12});
  require_allclose(cverl::distributed::shard_dim(x, 1, 1, 3), x.narrow(1, 4, 4), "shard dim");
  require_allclose(cverl::distributed::shard_dim(x, -1, 2, 3), x.narrow(1, 8, 4), "negative shard dim");

  require_throws([&]() { (void)cverl::distributed::shard_dim(x, 1, 0, 5); }, "non-divisible shard rejected");
  require_throws([&]() { (void)cverl::distributed::shard_dim(x, 2, 0, 1); }, "out-of-range shard dim rejected");
  require_throws([&]() { (void)cverl::distributed::shard_dim(x, 1, 3, 3); }, "out-of-range shard rank rejected");
}

void test_tp_linear_manual_reduce() {
  torch::manual_seed(1);
  auto x = torch::randn({2, 3, 8});
  auto w_col = torch::randn({12, 8});
  auto w_row = torch::randn({5, 8});

  std::vector<torch::Tensor> col_parts;
  std::vector<torch::Tensor> row_parts;
  for (int64_t rank = 0; rank < 4; ++rank) {
    cverl::distributed::ParallelGroup group{rank, 4, {0, 1, 2, 3}, nullptr};
    col_parts.push_back(cverl::distributed::column_parallel_linear(x, w_col, group));
    row_parts.push_back(cverl::distributed::row_parallel_linear(x, w_row, group, false));
  }
  require_allclose(torch::cat(col_parts, -1), dense_linear(x, w_col), "column parallel linear");
  require_allclose(torch::stack(row_parts).sum(0), dense_linear(x, w_row), "row parallel linear");
}

void test_tp_mlp_manual_reduce() {
  torch::manual_seed(2);
  auto x = torch::randn({2, 3, 8});
  auto gate = torch::randn({16, 8});
  auto up = torch::randn({16, 8});
  auto down = torch::randn({8, 16});
  auto dense = dense_linear(torch::silu(dense_linear(x, gate)) * dense_linear(x, up), down);

  std::vector<torch::Tensor> partials;
  for (int64_t rank = 0; rank < 4; ++rank) {
    cverl::distributed::ParallelGroup group{rank, 4, {0, 1, 2, 3}, nullptr};
    auto gate_shard = cverl::distributed::column_parallel_linear(x, gate, group);
    auto up_shard = cverl::distributed::column_parallel_linear(x, up, group);
    auto hidden = torch::silu(gate_shard) * up_shard;
    auto down_shard = cverl::distributed::shard_dim(down, 1, rank, 4);
    partials.push_back(dense_linear(hidden, down_shard));
  }
  require_allclose(torch::stack(partials).sum(0), dense, "tp mlp manual reduce", 1.0e-5, 2.0e-5);
}

void test_single_rank_tp_paths() {
  torch::manual_seed(3);
  cverl::distributed::SingleProcessCollectives collectives;
  cverl::distributed::ParallelGroup group{0, 1, {0}, &collectives};
  auto x = torch::randn({2, 3, 8});
  auto w_row = torch::randn({5, 8});
  auto gate = torch::randn({16, 8});
  auto up = torch::randn({16, 8});
  auto down = torch::randn({8, 16});

  require_allclose(cverl::distributed::row_parallel_linear(x, w_row, group, true), dense_linear(x, w_row),
                   "single-rank row parallel reduce");
  auto dense_mlp = dense_linear(torch::silu(dense_linear(x, gate)) * dense_linear(x, up), down);
  require_allclose(cverl::distributed::tensor_parallel_mlp_swiglu(x, gate, up, down, group), dense_mlp,
                   "single-rank tp mlp", 1.0e-5, 2.0e-5);
}

void test_parallel_group_validation() {
  auto x = torch::randn({2, 3, 8});
  auto w = torch::randn({4, 8});
  cverl::distributed::ParallelGroup bad_rank{2, 2, {0, 1}, nullptr};
  cverl::distributed::ParallelGroup bad_ranks{0, 2, {0}, nullptr};
  cverl::distributed::ParallelGroup needs_collectives{0, 2, {0, 1}, nullptr};
  require_throws([&]() { (void)cverl::distributed::column_parallel_linear(x, w, bad_rank); },
                 "invalid parallel rank rejected");
  require_throws([&]() { (void)cverl::distributed::column_parallel_linear(x, w, bad_ranks); },
                 "invalid group ranks rejected");
  require_throws([&]() { (void)cverl::distributed::row_parallel_linear(x, w, needs_collectives, true); },
                 "missing collectives rejected");
}

void test_dp_sync_single_process() {
  cverl::distributed::SingleProcessCollectives collectives;
  auto p = torch::randn({2, 3}, torch::requires_grad());
  auto loss = (p * p).sum();
  loss.backward();
  auto before = p.grad().clone();
  cverl::distributed::data_parallel_sync_gradients({p}, collectives, {0}, true);
  require_allclose(p.grad(), before, "single process dp sync");

  CountingCollectives counting;
  auto counted = torch::randn({2, 3}, torch::requires_grad());
  (counted * counted).sum().backward();
  auto counted_before = counted.grad().clone();
  cverl::distributed::data_parallel_sync_gradients({counted}, counting, {0}, true);
  if (counting.all_reduce_calls != 0) {
    throw std::runtime_error("single-rank dp sync should skip all_reduce");
  }
  require_allclose(counted.grad(), counted_before, "single-rank dp sync grad unchanged");

  auto no_grad = torch::randn({2, 3});
  auto no_backward = torch::randn({2, 3}, torch::requires_grad());
  cverl::distributed::data_parallel_sync_gradients({no_grad, no_backward}, collectives, {0}, false);
  if (no_backward.grad().defined()) {
    throw std::runtime_error("parameter without grad should stay untouched");
  }
}

void test_dp_sync_bucketed() {
  CountingCollectives collectives(2, 0);
  auto p0 = torch::randn({2, 3}, torch::requires_grad());
  auto p1 = torch::randn({4}, torch::requires_grad());
  auto p2 = torch::randn({5}, torch::requires_grad());
  auto loss = (p0 * p0).sum() + (p1 * 3).sum() + (p2 * -2).sum();
  loss.backward();
  auto g0 = p0.grad().clone();
  auto g1 = p1.grad().clone();
  auto g2 = p2.grad().clone();

  cverl::distributed::data_parallel_sync_gradients({p0, p1, p2}, collectives, {0, 1}, true, 1024 * 1024);
  if (collectives.all_reduce_calls != 1) {
    throw std::runtime_error("bucketed dp sync should use one all_reduce for same dtype/device");
  }
  if (collectives.last_all_reduce_numel != p0.numel() + p1.numel() + p2.numel()) {
    throw std::runtime_error("bucketed dp sync all_reduce numel mismatch");
  }
  require_allclose(p0.grad(), g0, "bucketed dp sync p0");
  require_allclose(p1.grad(), g1, "bucketed dp sync p1");
  require_allclose(p2.grad(), g2, "bucketed dp sync p2");

  collectives.all_reduce_calls = 0;
  cverl::distributed::data_parallel_sync_gradients({p0, p1, p2}, collectives, {0, 1}, true, 16);
  if (collectives.all_reduce_calls <= 1) {
    throw std::runtime_error("small dp sync bucket should split all_reduce calls");
  }
  if (collectives.last_all_reduce_numel != p2.numel()) {
    throw std::runtime_error("single-entry dp sync bucket should all_reduce only that tensor");
  }

  CountingCollectives outside_rank(3, 2);
  require_throws([&]() {
    cverl::distributed::data_parallel_sync_gradients({p0}, outside_rank, {0, 1}, true, 1024);
  }, "multi-rank dp sync should reject rank outside data group before collectives");
}

void test_dp_sync_forced_communication_dtype() {
  CountingCollectives collectives(2, 0);
  auto p0 = torch::ones({4}, torch::TensorOptions().dtype(torch::kBFloat16).requires_grad(true));
  auto p1 = torch::full({4}, 2.0, torch::TensorOptions().dtype(torch::kBFloat16).requires_grad(true));
  p0.mutable_grad() = torch::full({4}, 0.25, torch::TensorOptions().dtype(torch::kBFloat16));
  p1.mutable_grad() = torch::full({4}, -0.5, torch::TensorOptions().dtype(torch::kBFloat16));

  cverl::distributed::data_parallel_sync_gradients(
      {p0, p1}, collectives, {0, 1}, true, 1024, torch::kFloat32);
  if (collectives.all_reduce_calls != 1 || collectives.last_all_reduce_dtype != torch::kFloat32) {
    throw std::runtime_error("forced DP all-reduce dtype should use one fp32 bucket");
  }
  if (p0.grad().scalar_type() != torch::kBFloat16 || p1.grad().scalar_type() != torch::kBFloat16) {
    throw std::runtime_error("forced DP all-reduce should preserve original grad dtype");
  }
  require_allclose(p0.grad().to(torch::kFloat32),
                   torch::full({4}, 0.25, torch::kFloat32),
                   "forced dtype dp sync p0");
  require_allclose(p1.grad().to(torch::kFloat32),
                   torch::full({4}, -0.5, torch::kFloat32),
                   "forced dtype dp sync p1");

  collectives.reduce_scatter_calls = 0;
  auto buckets = cverl::distributed::data_parallel_reduce_scatter_gradients(
      {p0, p1}, collectives, {0, 1}, true, 1024, torch::kFloat32);
  if (collectives.reduce_scatter_calls != 1 ||
      collectives.last_reduce_scatter_dtype != torch::kFloat32 ||
      buckets.size() != 1 || buckets[0].shard.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("forced DP reduce-scatter dtype should produce one fp32 shard");
  }
  require_allclose(buckets[0].shard,
                   torch::cat({torch::full({4}, 0.25, torch::kFloat32),
                               torch::full({4}, -0.5, torch::kFloat32)}, 0)
                       .narrow(0, 0, 4),
                   "forced dtype reduce-scatter shard");
}

void test_dp_reduce_scatter_bucketed_single_process() {
  CountingCollectives collectives;
  auto p0 = torch::randn({2, 3}, torch::requires_grad());
  auto p1 = torch::randn({4}, torch::requires_grad());
  auto p2 = torch::randn({5}, torch::requires_grad());
  auto loss = (p0 * 2).sum() + (p1 * -3).sum() + (p2 * p2).sum();
  loss.backward();
  auto g0 = p0.grad().clone().view({-1});
  auto g1 = p1.grad().clone().view({-1});
  auto g2 = p2.grad().clone().view({-1});

  auto buckets = cverl::distributed::data_parallel_reduce_scatter_gradients(
      {p0, p1, p2}, collectives, {0}, true, 1024 * 1024);
  if (collectives.reduce_scatter_calls != 0 || buckets.size() != 1) {
    throw std::runtime_error("single-rank reduce-scatter gradients should skip collectives");
  }
  if (collectives.last_reduce_scatter_numel != 0) {
    throw std::runtime_error("reduce-scatter gradient bucket should keep default last numel");
  }
  require_allclose(buckets[0].shard, torch::cat({g0, g1, g2}, 0), "reduce-scatter flat shard");
  if (buckets[0].original_numel != g0.numel() + g1.numel() + g2.numel() ||
      buckets[0].padded_numel != buckets[0].original_numel) {
    throw std::runtime_error("reduce-scatter bucket numel metadata mismatch");
  }
  if (buckets[0].parameter_indices != std::vector<int64_t>{0, 1, 2} ||
      buckets[0].parameter_numels != std::vector<int64_t>{g0.numel(), g1.numel(), g2.numel()}) {
    throw std::runtime_error("reduce-scatter bucket parameter metadata mismatch");
  }

  collectives.reduce_scatter_calls = 0;
  buckets = cverl::distributed::data_parallel_reduce_scatter_gradients(
      {p0, p1, p2}, collectives, {0}, true, 16);
  if (collectives.reduce_scatter_calls != 0 || buckets.size() <= 1) {
    throw std::runtime_error("single-rank small reduce-scatter bucket should split metadata without collectives");
  }
  if (collectives.last_reduce_scatter_numel != 0) {
    throw std::runtime_error("single-rank reduce-scatter should keep default last numel");
  }

  CountingCollectives outside_rank(3, 2);
  require_throws([&]() {
    (void)cverl::distributed::data_parallel_reduce_scatter_gradients(
        {p0}, outside_rank, {0, 1}, true, 1024);
  }, "multi-rank reduce-scatter should reject rank outside data group before collectives");
}

void test_single_process_collectives_validation() {
  cverl::distributed::SingleProcessCollectives collectives;
  auto x = torch::ones({2, 2});
  require_throws([&]() { (void)collectives.all_reduce(x, cverl::distributed::ReduceOp::Sum, {0, 1}); },
                 "single collectives invalid group rejected");
  require_throws([&]() { collectives.send(x, 1); }, "single collectives invalid send rejected");
  require_throws([&]() { (void)collectives.recv_like(x, 1); }, "single collectives invalid recv rejected");
}

}  // namespace

int main() {
  try {
    test_shard_dim();
    test_tp_linear_manual_reduce();
    test_tp_mlp_manual_reduce();
    test_single_rank_tp_paths();
    test_parallel_group_validation();
    test_dp_sync_single_process();
    test_dp_sync_bucketed();
    test_dp_sync_forced_communication_dtype();
    test_dp_reduce_scatter_bucketed_single_process();
    test_single_process_collectives_validation();
  } catch (const std::exception& e) {
    std::cerr << "test_parallel_ops failed: " << e.what() << "\n";
    return 1;
  }
  std::cout << "distributed parallel ops tests passed\n";
  return 0;
}
