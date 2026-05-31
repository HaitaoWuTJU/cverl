#include "cverl/distributed/optimizer_sharding.h"

#include <iostream>
#include <stdexcept>
#include <vector>

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

}  // namespace

int main() {
  try {
    test_greedy_owners_cover_every_parameter();
    test_owned_indices();
    test_validation();
    std::cout << "optimizer sharding tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_optimizer_sharding failed: " << e.what() << "\n";
    return 1;
  }
}
