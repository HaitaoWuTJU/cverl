#include "cverl/distributed/optimizer_sharding.h"

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

}  // namespace

int main() {
  try {
    test_greedy_owners_cover_every_parameter();
    test_owned_indices();
    test_validation();
    test_flat_parameter_shards_roundtrip();
    test_flat_parameter_shard_update_slice();
    std::cout << "optimizer sharding tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_optimizer_sharding failed: " << e.what() << "\n";
    return 1;
  }
}
