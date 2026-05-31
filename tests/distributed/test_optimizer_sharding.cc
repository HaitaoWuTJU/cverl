#include "cverl/distributed/optimizer_sharding.h"
#include "cverl/torch/fp32_master_adamw.h"

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <torch/torch.h>

namespace {

class SliceCollectives final : public cverl::distributed::Collectives {
 public:
  explicit SliceCollectives(int64_t rank) : rank_(rank) {}
  SliceCollectives(int64_t rank, std::vector<torch::Tensor> gather_shards)
      : rank_(rank), gather_shards_(std::move(gather_shards)) {}

  int64_t rank() const override { return rank_; }
  int64_t world_size() const override { return 2; }
  void barrier() override {}

  torch::Tensor broadcast(const torch::Tensor& input,
                          int64_t /*root*/,
                          const std::vector<int64_t>& /*group*/) override {
    return input.clone();
  }

  torch::Tensor all_reduce(const torch::Tensor& input,
                           cverl::distributed::ReduceOp /*op*/,
                           const std::vector<int64_t>& /*group*/) override {
    ++all_reduce_calls;
    return input.clone();
  }

  torch::Tensor all_gather(const torch::Tensor& input,
                           const std::vector<int64_t>& group,
                           int64_t dim) override {
    if (dim != 0) {
      throw std::invalid_argument("SliceCollectives all_gather dim must be 0");
    }
    ++all_gather_calls;
    if (!gather_shards_.empty()) {
      if (gather_shards_.size() != group.size()) {
        throw std::invalid_argument("SliceCollectives gather shard count mismatch");
      }
      std::vector<torch::Tensor> shards;
      shards.reserve(gather_shards_.size());
      for (const auto& shard : gather_shards_) {
        if (all_gather_offset_ + input.numel() > shard.numel()) {
          throw std::invalid_argument("SliceCollectives gather shard slice out of range");
        }
        shards.push_back(shard.narrow(0, all_gather_offset_, input.numel()).clone());
      }
      all_gather_offset_ += input.numel();
      return torch::cat(shards, 0);
    }
    std::vector<torch::Tensor> copies;
    copies.reserve(group.size());
    for (size_t i = 0; i < group.size(); ++i) {
      copies.push_back(input.clone());
    }
    return torch::cat(copies, 0);
  }

  torch::Tensor reduce_scatter(const torch::Tensor& input,
                               cverl::distributed::ReduceOp op,
                               const std::vector<int64_t>& group,
                               int64_t dim) override {
    if (dim != 0 || group.size() != 2 || input.size(0) % 2 != 0) {
      throw std::invalid_argument("SliceCollectives reduce_scatter expects two even shards");
    }
    last_reduce_op = op;
    ++reduce_scatter_calls;
    const int64_t shard = input.size(0) / 2;
    return input.narrow(0, rank_ * shard, shard).clone();
  }

  void send(const torch::Tensor& /*input*/, int64_t /*peer*/) override {}
  torch::Tensor recv_like(const torch::Tensor& like, int64_t /*peer*/) override {
    return torch::empty_like(like);
  }

  int64_t reduce_scatter_calls = 0;
  int64_t all_gather_calls = 0;
  int64_t all_reduce_calls = 0;
  cverl::distributed::ReduceOp last_reduce_op = cverl::distributed::ReduceOp::Sum;

 private:
  int64_t rank_ = 0;
  int64_t all_gather_offset_ = 0;
  std::vector<torch::Tensor> gather_shards_;
};

void require(bool cond, const char* msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

void require_near(double actual, double expected, double atol, const char* msg) {
  if (std::abs(actual - expected) > atol) {
    std::cerr << msg << "\nactual=" << actual << "\nexpected=" << expected << "\n";
    throw std::runtime_error(msg);
  }
}

template <typename Fn>
void require_throws(Fn&& fn, const char* msg) {
  bool threw = false;
  try {
    fn();
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  if (!threw) {
    throw std::runtime_error(msg);
  }
}

void test_greedy_owners_cover_every_parameter() {
  const std::vector<int64_t> bytes{100, 60, 50, 40, 30, 20};
  auto owners = cverl::distributed::greedy_parameter_owner_by_size(bytes, 3);
  require(owners.size() == bytes.size(), "owner size mismatch");
  std::vector<int64_t> counts(3, 0);
  std::vector<int64_t> sums(3, 0);
  for (size_t i = 0; i < owners.size(); ++i) {
    require(owners[i] >= 0 && owners[i] < 3, "owner out of range");
    counts[static_cast<size_t>(owners[i])] += 1;
    sums[static_cast<size_t>(owners[i])] += bytes[i];
  }
  require(counts[0] + counts[1] + counts[2] == static_cast<int64_t>(bytes.size()), "owner coverage mismatch");
  require(sums[0] == 100 && sums[1] == 110 && sums[2] == 90, "unexpected greedy byte balance");
}

void test_greedy_owners_sort_by_size_before_assignment() {
  const std::vector<int64_t> bytes{1, 1, 100, 1, 1, 100};
  auto owners = cverl::distributed::greedy_parameter_owner_by_size(bytes, 2);
  require(owners.size() == bytes.size(), "unordered owner size mismatch");
  require(owners[2] != owners[5], "largest parameters should split across ranks");
  std::vector<int64_t> sums(2, 0);
  for (size_t i = 0; i < bytes.size(); ++i) {
    require(owners[i] >= 0 && owners[i] < 2, "unordered owner out of range");
    sums[static_cast<size_t>(owners[i])] += bytes[i];
  }
  require(sums[0] == 102 && sums[1] == 102, "size-sorted greedy owner balance");
}

void test_owned_indices() {
  const std::vector<int64_t> owners{0, 1, 2, 1, 0};
  auto rank0 = cverl::distributed::owned_parameter_indices(owners, 0);
  auto rank1 = cverl::distributed::owned_parameter_indices(owners, 1);
  auto rank2 = cverl::distributed::owned_parameter_indices(owners, 2);
  require((rank0 == std::vector<int64_t>{0, 4}), "rank0 owned indices");
  require((rank1 == std::vector<int64_t>{1, 3}), "rank1 owned indices");
  require((rank2 == std::vector<int64_t>{2}), "rank2 owned indices");
}

void test_validation() {
  require_throws([&]() { (void)cverl::distributed::greedy_parameter_owner_by_size({1}, 0); },
                 "invalid dp rejected");
  require_throws([&]() { (void)cverl::distributed::greedy_parameter_owner_by_size({-1}, 1); },
                 "negative bytes rejected");
  require_throws([&]() { (void)cverl::distributed::owned_parameter_indices({0, -1}, 0); },
                 "negative owner rejected");
}

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

void test_flat_parameter_shards_roundtrip() {
  auto p0 = torch::arange(1, 7, torch::TensorOptions().dtype(torch::kFloat32)).reshape({2, 3}).contiguous();
  auto p1 = torch::arange(11, 15, torch::TensorOptions().dtype(torch::kFloat32)).contiguous();
  auto p2 = torch::arange(21, 26, torch::TensorOptions().dtype(torch::kFloat32)).contiguous();
  std::vector<torch::Tensor> params{p0, p1, p2};

  auto rank0 = cverl::distributed::flatten_parameter_shard(params, 2, 0);
  auto rank1 = cverl::distributed::flatten_parameter_shard(params, 2, 1);
  require(rank0.original_numel == 15 && rank0.padded_numel == 16, "rank0 shard metadata");
  require(rank1.original_numel == 15 && rank1.padded_numel == 16, "rank1 shard metadata");
  require(rank0.shard.numel() == 8 && rank1.shard.numel() == 8, "shard sizes");
  require(rank0.ranges.size() == 2, "rank0 ranges should cover p0 and p1");
  require(rank1.ranges.size() == 2, "rank1 ranges should cover p1 and p2");

  auto target0 = torch::zeros_like(p0);
  auto target1 = torch::zeros_like(p1);
  auto target2 = torch::zeros_like(p2);
  std::vector<torch::Tensor> target{target0, target1, target2};
  cverl::distributed::apply_flat_parameter_shard(rank0, target);
  cverl::distributed::apply_flat_parameter_shard(rank1, target);
  require_allclose(target0, p0, "roundtrip p0");
  require_allclose(target1, p1, "roundtrip p1");
  require_allclose(target2, p2, "roundtrip p2");

  auto gathered = torch::cat({rank0.shard, rank1.shard}, 0).contiguous();
  auto full0 = torch::zeros_like(p0);
  auto full1 = torch::zeros_like(p1);
  auto full2 = torch::zeros_like(p2);
  std::vector<torch::Tensor> full_target{full0, full1, full2};
  cverl::distributed::apply_full_flat_parameters(gathered, rank0.original_numel, full_target);
  require_allclose(full0, p0, "full flat p0");
  require_allclose(full1, p1, "full flat p1");
  require_allclose(full2, p2, "full flat p2");
}

void test_single_tensor_flat_parameter_shards() {
  auto p = torch::arange(1, 10, torch::TensorOptions().dtype(torch::kFloat32)).contiguous();
  std::vector<torch::Tensor> params{p};

  auto rank0 = cverl::distributed::flatten_parameter_shard(params, 2, 0);
  auto rank1 = cverl::distributed::flatten_parameter_shard(params, 2, 1);
  require(rank0.original_numel == 9 && rank0.padded_numel == 10, "single tensor rank0 metadata");
  require(rank1.original_numel == 9 && rank1.padded_numel == 10, "single tensor rank1 metadata");
  require(rank0.ranges.size() == 1 && rank0.ranges[0].parameter_index == 0,
          "single tensor rank0 range metadata");
  require(rank1.ranges.size() == 1 && rank1.ranges[0].parameter_index == 0,
          "single tensor rank1 range metadata");
  require_allclose(rank0.shard, p.narrow(0, 0, 5), "single tensor rank0 shard");
  require_allclose(rank1.shard,
                   torch::cat({p.narrow(0, 5, 4), torch::zeros({1}, torch::kFloat32)}, 0),
                   "single tensor rank1 padded shard");

  auto target = torch::zeros_like(p);
  std::vector<torch::Tensor> target_params{target};
  cverl::distributed::apply_flat_parameter_shard(rank0, target_params);
  cverl::distributed::apply_flat_parameter_shard(rank1, target_params);
  require_allclose(target, p, "single tensor shard roundtrip");
}

void test_flat_parameter_shard_update_slice() {
  auto p0 = torch::zeros({3}, torch::TensorOptions().dtype(torch::kFloat32));
  auto p1 = torch::zeros({3}, torch::TensorOptions().dtype(torch::kFloat32));
  std::vector<torch::Tensor> params{p0, p1};
  auto shard = cverl::distributed::flatten_parameter_shard(params, 2, 1);
  shard.shard.fill_(5.0);
  cverl::distributed::apply_flat_parameter_shard(shard, params);
  require_allclose(p0, torch::zeros_like(p0), "rank1 should not touch p0");
  require_allclose(p1, torch::full_like(p1, 5.0), "rank1 should update p1");
}

void test_all_gather_apply_flat_parameter_shard() {
  auto p0 = torch::arange(1, 7, torch::TensorOptions().dtype(torch::kFloat32)).reshape({2, 3}).contiguous();
  auto p1 = torch::arange(11, 16, torch::TensorOptions().dtype(torch::kFloat32)).contiguous();
  std::vector<torch::Tensor> params{p0, p1};
  auto rank0 = cverl::distributed::flatten_parameter_shard(params, 2, 0);
  auto rank1 = cverl::distributed::flatten_parameter_shard(params, 2, 1);
  auto update0 = rank0.shard + 100.0;
  auto update1 = rank1.shard + 200.0;
  rank1.shard.copy_(update1);

  auto target0 = torch::zeros_like(p0);
  auto target1 = torch::zeros_like(p1);
  std::vector<torch::Tensor> target{target0, target1};
  SliceCollectives rank1_collectives(1, {update0, update1});
  cverl::distributed::all_gather_apply_flat_parameter_shard(rank1, rank1_collectives, {0, 1}, target);
  auto gathered = torch::cat({update0, update1}, 0).contiguous();
  auto expected0 = gathered.narrow(0, 0, p0.numel()).view(p0.sizes());
  auto expected1 = gathered.narrow(0, p0.numel(), p1.numel()).view(p1.sizes());
  require(rank1_collectives.all_gather_calls == 1, "flat all-gather apply should issue one all_gather");
  require_allclose(target0, expected0, "flat all-gather apply p0");
  require_allclose(target1, expected1, "flat all-gather apply p1");

  auto bucket_target0 = torch::zeros_like(p0);
  auto bucket_target1 = torch::zeros_like(p1);
  std::vector<torch::Tensor> bucket_target{bucket_target0, bucket_target1};
  SliceCollectives bucket_rank1_collectives(1, {update0, update1});
  cverl::distributed::all_gather_apply_flat_parameter_shard_bucketed(
      rank1, bucket_rank1_collectives, {0, 1}, bucket_target, 4);
  require(bucket_rank1_collectives.all_gather_calls == 3,
          "bucketed flat all-gather apply should issue one all_gather per bucket");
  require_allclose(bucket_target0, expected0, "bucketed flat all-gather apply p0");
  require_allclose(bucket_target1, expected1, "bucketed flat all-gather apply p1");

  auto single_rank = cverl::distributed::flatten_parameter_shard(params, 1, 0);
  single_rank.shard.add_(300.0);
  auto single_target0 = torch::zeros_like(p0);
  auto single_target1 = torch::zeros_like(p1);
  std::vector<torch::Tensor> single_target{single_target0, single_target1};
  SliceCollectives single_collectives(0);
  cverl::distributed::all_gather_apply_flat_parameter_shard(single_rank, single_collectives, {0}, single_target);
  require(single_collectives.all_gather_calls == 0, "single-rank flat all-gather apply should skip all_gather");
  require_allclose(single_target0, p0 + 300.0, "single-rank flat apply p0");
  require_allclose(single_target1, p1 + 300.0, "single-rank flat apply p1");

  auto bad = rank1;
  bad.shard_begin = 0;
  require_throws([&]() {
    (void)cverl::distributed::all_gather_flat_parameter_shards(bad, rank1_collectives, {0, 1});
  }, "bad flat shard metadata rejected");
}

void test_apply_full_flat_validation() {
  auto p = torch::zeros({3}, torch::TensorOptions().dtype(torch::kFloat32));
  require_throws([&]() {
    cverl::distributed::apply_full_flat_parameters(torch::zeros({4}, torch::kFloat32), 4, {p});
  }, "mismatched original numel rejected");
  require_throws([&]() {
    cverl::distributed::apply_full_flat_parameters(torch::zeros({4}, torch::kFloat32), 5, {p});
  }, "original numel beyond flat rejected");
}

void test_flat_gradient_shards() {
  auto p0 = torch::zeros({2, 3}, torch::TensorOptions().dtype(torch::kBFloat16));
  auto p1 = torch::zeros({5}, torch::TensorOptions().dtype(torch::kBFloat16));
  p0.set_requires_grad(true);
  p1.set_requires_grad(true);
  p0.mutable_grad() = torch::arange(1, 7, torch::TensorOptions().dtype(torch::kFloat32)).reshape({2, 3});
  p1.mutable_grad() = torch::arange(11, 16, torch::TensorOptions().dtype(torch::kFloat32));
  std::vector<torch::Tensor> params{p0, p1};

  auto rank0 = cverl::distributed::flatten_gradient_shard(params, 2, 0);
  auto rank1 = cverl::distributed::flatten_gradient_shard(params, 2, 1);
  auto gathered = torch::cat({rank0.shard, rank1.shard}, 0).contiguous();
  auto expected = torch::cat({p0.grad().to(torch::kFloat32).view({-1}),
                              p1.grad().to(torch::kFloat32).view({-1}),
                              torch::zeros({1}, torch::kFloat32)}, 0).contiguous();
  require_allclose(gathered, expected, "flat gradient shards should match parameter layout");

  auto no_grad = torch::zeros({3}, torch::TensorOptions().dtype(torch::kFloat32));
  no_grad.set_requires_grad(true);
  require_throws([&]() {
    (void)cverl::distributed::flatten_gradient_shard({no_grad}, 1, 0);
  }, "missing gradient rejected");
  auto optional = cverl::distributed::flatten_gradient_shard({no_grad}, 1, 0, false);
  require_allclose(optional.shard, torch::zeros({3}, torch::kFloat32), "optional missing gradient zeros");
}

void test_reduce_scatter_flat_gradient_shard() {
  auto p0 = torch::zeros({2, 3}, torch::TensorOptions().dtype(torch::kBFloat16));
  auto p1 = torch::zeros({5}, torch::TensorOptions().dtype(torch::kBFloat16));
  p0.set_requires_grad(true);
  p1.set_requires_grad(true);
  p0.mutable_grad() = torch::arange(1, 7, torch::TensorOptions().dtype(torch::kFloat32)).reshape({2, 3});
  p1.mutable_grad() = torch::arange(11, 16, torch::TensorOptions().dtype(torch::kFloat32));
  std::vector<torch::Tensor> params{p0, p1};

  SliceCollectives rank0(0);
  SliceCollectives rank1(1);
  auto shard0 = cverl::distributed::reduce_scatter_flat_gradient_shard(params, rank0, {0, 1}, true);
  auto shard1 = cverl::distributed::reduce_scatter_flat_gradient_shard(params, rank1, {0, 1}, false);
  auto expected = torch::cat({p0.grad().to(torch::kFloat32).view({-1}),
                              p1.grad().to(torch::kFloat32).view({-1}),
                              torch::zeros({1}, torch::kFloat32)}, 0).contiguous();
  require(rank0.reduce_scatter_calls == 1 && rank1.reduce_scatter_calls == 1,
          "flat reduce-scatter should issue one collective per rank");
  require(rank0.last_reduce_op == cverl::distributed::ReduceOp::Mean, "average uses mean reduce-scatter");
  require(rank1.last_reduce_op == cverl::distributed::ReduceOp::Sum, "non-average uses sum reduce-scatter");
  require(shard0.original_numel == 11 && shard0.padded_numel == 12, "rank0 flat reduce-scatter metadata");
  require(shard1.original_numel == 11 && shard1.padded_numel == 12, "rank1 flat reduce-scatter metadata");
  require(shard0.ranges.size() == 1 && shard0.ranges[0].parameter_index == 0,
          "rank0 flat reduce-scatter ranges should be metadata-only accurate");
  require(shard1.ranges.size() == 1 && shard1.ranges[0].parameter_index == 1,
          "rank1 flat reduce-scatter ranges should be metadata-only accurate");
  require_allclose(shard0.shard, expected.narrow(0, 0, 6), "rank0 flat reduce-scatter shard");
  require_allclose(shard1.shard, expected.narrow(0, 6, 6), "rank1 flat reduce-scatter shard");

  SliceCollectives bucket_rank0(0);
  SliceCollectives bucket_rank1(1);
  auto bucket0 = cverl::distributed::reduce_scatter_flat_gradient_shard_bucketed(
      params, bucket_rank0, {0, 1}, false, 4);
  auto bucket1 = cverl::distributed::reduce_scatter_flat_gradient_shard_bucketed(
      params, bucket_rank1, {0, 1}, false, 4);
  require(bucket_rank0.reduce_scatter_calls == 3 && bucket_rank1.reduce_scatter_calls == 3,
          "bucketed flat reduce-scatter should split by bucket");
  require_allclose(bucket0.shard, expected.narrow(0, 0, 6), "rank0 bucketed flat reduce-scatter shard");
  require_allclose(bucket1.shard, expected.narrow(0, 6, 6), "rank1 bucketed flat reduce-scatter shard");

  SliceCollectives bad_rank(1);
  require_throws([&]() {
    (void)cverl::distributed::reduce_scatter_flat_gradient_shard(params, bad_rank, {0}, true);
  }, "rank outside data group rejected");
}

void test_flat_sharded_adamw_matches_dense() {
  cverl::torch_backend::Fp32MasterAdamWOptions opts;
  opts.lr = 2.0e-4;
  opts.beta1 = 0.9;
  opts.beta2 = 0.95;
  opts.eps = 1.0e-8;
  opts.weight_decay = 0.01;
  opts.use_master_weights = true;

  auto dense_p0 = torch::linspace(-1.0, 1.0, 7, torch::TensorOptions().dtype(torch::kBFloat16));
  auto dense_p1 = torch::linspace(2.0, 3.0, 4, torch::TensorOptions().dtype(torch::kBFloat16));
  dense_p0.set_requires_grad(true);
  dense_p1.set_requires_grad(true);
  auto sharded_p0 = dense_p0.detach().clone();
  auto sharded_p1 = dense_p1.detach().clone();
  sharded_p0.set_requires_grad(true);
  sharded_p1.set_requires_grad(true);

  auto grad0 = torch::linspace(0.01, 0.07, 7, torch::TensorOptions().dtype(torch::kFloat32));
  auto grad1 = torch::linspace(-0.04, -0.01, 4, torch::TensorOptions().dtype(torch::kFloat32));
  dense_p0.mutable_grad() = grad0.to(dense_p0.scalar_type());
  dense_p1.mutable_grad() = grad1.to(dense_p1.scalar_type());
  sharded_p0.mutable_grad() = grad0.to(sharded_p0.scalar_type());
  sharded_p1.mutable_grad() = grad1.to(sharded_p1.scalar_type());

  std::vector<torch::Tensor> dense_params{dense_p0, dense_p1};
  std::vector<torch::Tensor> sharded_params{sharded_p0, sharded_p1};
  cverl::torch_backend::Fp32MasterAdamW dense_optimizer(dense_params, opts);
  dense_optimizer.accumulate_model_grads();
  dense_optimizer.step();

  auto param_rank0 = cverl::distributed::flatten_parameter_shard(sharded_params, 2, 0);
  auto param_rank1 = cverl::distributed::flatten_parameter_shard(sharded_params, 2, 1);
  auto grad_rank0 = cverl::distributed::flatten_gradient_shard(sharded_params, 2, 0);
  auto grad_rank1 = cverl::distributed::flatten_gradient_shard(sharded_params, 2, 1);
  require(param_rank0.original_numel == grad_rank0.original_numel, "rank0 original numel metadata");
  require(param_rank0.padded_numel == grad_rank0.padded_numel, "rank0 padded numel metadata");
  require(param_rank1.original_numel == grad_rank1.original_numel, "rank1 original numel metadata");
  require(param_rank1.padded_numel == grad_rank1.padded_numel, "rank1 padded numel metadata");

  cverl::torch_backend::FlatAdamW flat_rank0(param_rank0.shard, opts);
  cverl::torch_backend::FlatAdamW flat_rank1(param_rank1.shard, opts);
  SliceCollectives comm0(0);
  SliceCollectives comm1(1);
  auto step0 = cverl::distributed::flat_sharded_adamw_step(
      sharded_params, param_rank0, flat_rank0, comm0, {0, 1}, comm0, {0, 1}, 0.0, false, true, false, 4);
  auto step1 = cverl::distributed::flat_sharded_adamw_step(
      sharded_params, param_rank1, flat_rank1, comm1, {0, 1}, comm1, {0, 1}, 0.0, false, true, false, 4);
  require(step0.gradient_shard.shard.numel() == param_rank0.shard.numel(), "rank0 flat step grad shard");
  require(step1.gradient_shard.shard.numel() == param_rank1.shard.numel(), "rank1 flat step grad shard");
  require(step0.grad_clip_scale == 1.0 && step1.grad_clip_scale == 1.0, "flat step unclipped");
  require(step0.local_grad_norm > 0.0 && step1.local_grad_norm > 0.0, "flat step grad norms");
  require(comm0.reduce_scatter_calls == 3 && comm1.reduce_scatter_calls == 3,
          "flat sharded AdamW bucketed step should reduce-scatter once per bucket per rank");
  auto gathered =
      torch::cat({flat_rank0.parameter_shard(), flat_rank1.parameter_shard()}, 0).contiguous();
  cverl::distributed::apply_full_flat_parameters(gathered, param_rank0.original_numel, sharded_params);

  require_allclose(sharded_p0.to(torch::kFloat32), dense_p0.to(torch::kFloat32),
                   "flat sharded AdamW p0 matches dense model", 1.0e-5, 2.0e-5);
  require_allclose(sharded_p1.to(torch::kFloat32), dense_p1.to(torch::kFloat32),
                   "flat sharded AdamW p1 matches dense model", 1.0e-5, 2.0e-5);

  auto flat_dense_master = torch::cat({dense_optimizer.master_parameters()[0].view({-1}),
                                       dense_optimizer.master_parameters()[1].view({-1})}, 0);
  require_allclose(gathered.narrow(0, 0, param_rank0.original_numel),
                   flat_dense_master,
                   "flat sharded AdamW fp32 master matches dense master",
                   1.0e-5,
                   2.0e-5);
}

void test_flat_sharded_adamw_clipped_local_norm() {
  cverl::torch_backend::Fp32MasterAdamWOptions opts;
  opts.lr = 1.0e-4;
  opts.beta1 = 0.9;
  opts.beta2 = 0.95;
  opts.eps = 1.0e-8;
  opts.weight_decay = 0.0;
  opts.use_master_weights = true;

  auto p = torch::zeros({4}, torch::TensorOptions().dtype(torch::kBFloat16));
  p.set_requires_grad(true);
  p.mutable_grad() = torch::full({4}, 4.0, torch::TensorOptions().dtype(torch::kFloat32));
  std::vector<torch::Tensor> params{p};

  auto param_rank0 = cverl::distributed::flatten_parameter_shard(params, 2, 0);
  cverl::torch_backend::FlatAdamW flat_rank0(param_rank0.shard, opts);
  SliceCollectives comm0(0);
  const double max_grad_norm = 1.0;
  auto step = cverl::distributed::flat_sharded_adamw_step(
      params, param_rank0, flat_rank0, comm0, {0, 1}, comm0, {0, 1}, max_grad_norm, false, true, false, 0);
  require(step.grad_clip_scale < 1.0, "flat clipped step should reduce gradients");
  require_near(step.local_grad_norm,
               std::sqrt(step.local_grad_norm_sq) * step.grad_clip_scale,
               1.0e-6,
               "flat clipped local norm should come from unclipped norm and scale");
  require_near(step.local_grad_norm, max_grad_norm, 1.0e-5, "flat clipped local norm should equal max norm");
}

void test_flat_sharded_adamw_single_dp_skips_data_collectives() {
  cverl::torch_backend::Fp32MasterAdamWOptions opts;
  opts.lr = 1.0e-4;
  opts.beta1 = 0.9;
  opts.beta2 = 0.95;
  opts.eps = 1.0e-8;
  opts.weight_decay = 0.0;
  opts.use_master_weights = true;

  auto p0 = torch::linspace(-1.0, 1.0, 5, torch::TensorOptions().dtype(torch::kBFloat16));
  auto p1 = torch::linspace(2.0, 3.0, 3, torch::TensorOptions().dtype(torch::kBFloat16));
  p0.set_requires_grad(true);
  p1.set_requires_grad(true);
  p0.mutable_grad() = torch::full({5}, 0.25, torch::TensorOptions().dtype(torch::kFloat32));
  p1.mutable_grad() = torch::full({3}, -0.5, torch::TensorOptions().dtype(torch::kFloat32));
  std::vector<torch::Tensor> params{p0, p1};

  auto param_rank0 = cverl::distributed::flatten_parameter_shard(params, 1, 0);
  cverl::torch_backend::FlatAdamW flat_rank0(param_rank0.shard, opts);
  SliceCollectives comm0(0);
  auto step = cverl::distributed::flat_sharded_adamw_step(
      params, param_rank0, flat_rank0, comm0, {0}, comm0, {0}, 0.0, true, true, true, 4, 4);
  require(step.gradient_shard.shard.numel() == param_rank0.shard.numel(), "single-DP grad shard size");
  require(comm0.reduce_scatter_calls == 0, "single-DP flat step should skip reduce-scatter");
  require(comm0.all_gather_calls == 0, "single-DP flat step should skip parameter all-gather");
  require(comm0.all_reduce_calls == 0, "single-DP flat step should skip norm all-reduce");
}

}  // namespace

int main() {
  try {
    test_greedy_owners_cover_every_parameter();
    test_greedy_owners_sort_by_size_before_assignment();
    test_owned_indices();
    test_validation();
    test_flat_parameter_shards_roundtrip();
    test_single_tensor_flat_parameter_shards();
    test_flat_parameter_shard_update_slice();
    test_all_gather_apply_flat_parameter_shard();
    test_apply_full_flat_validation();
    test_flat_gradient_shards();
    test_reduce_scatter_flat_gradient_shard();
    test_flat_sharded_adamw_matches_dense();
    test_flat_sharded_adamw_clipped_local_norm();
    test_flat_sharded_adamw_single_dp_skips_data_collectives();
    std::cout << "optimizer sharding tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_optimizer_sharding failed: " << e.what() << "\n";
    return 1;
  }
}
