#include "cverl/distributed/weight_sync.h"

#include <torch/torch.h>

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

class CountingCollectives final : public cverl::distributed::Collectives {
 public:
  int64_t broadcast_calls = 0;
  int64_t barrier_calls = 0;
  int64_t last_broadcast_numel = 0;

  int64_t rank() const override { return 0; }
  int64_t world_size() const override { return 2; }
  void barrier() override { ++barrier_calls; }

  torch::Tensor broadcast(const torch::Tensor& input,
                          int64_t root,
                          const std::vector<int64_t>& group) override {
    require_single_root(root, group);
    ++broadcast_calls;
    last_broadcast_numel = input.numel();
    return input.clone();
  }

  torch::Tensor all_reduce(const torch::Tensor& input,
                           cverl::distributed::ReduceOp /*op*/,
                           const std::vector<int64_t>& group) override {
    require_single_root(0, group);
    return input.clone();
  }

  torch::Tensor all_gather(const torch::Tensor& input,
                           const std::vector<int64_t>& group,
                           int64_t /*dim*/) override {
    require_single_root(0, group);
    return input.clone();
  }

  torch::Tensor reduce_scatter(const torch::Tensor& input,
                               cverl::distributed::ReduceOp /*op*/,
                               const std::vector<int64_t>& group,
                               int64_t /*dim*/) override {
    require_single_root(0, group);
    return input.clone();
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
  static void require_single_root(int64_t root, const std::vector<int64_t>& group) {
    bool have_rank = false;
    bool have_root = false;
    for (int64_t member : group) {
      have_rank = have_rank || member == 0;
      have_root = have_root || member == root;
    }
    if (!have_rank || !have_root) {
      throw std::invalid_argument("CountingCollectives requires rank/root in group");
    }
  }
};

void require(bool cond, const std::string& msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

struct TinyModule : torch::nn::Module {
  TinyModule() {
    w = register_parameter("w", torch::arange(6, torch::kFloat32).reshape({2, 3}).contiguous());
    b = register_parameter("b", torch::ones({3}, torch::kFloat32).contiguous());
  }
  torch::Tensor w;
  torch::Tensor b;
};

}  // namespace

int main() {
  try {
    TinyModule module;
    cverl::distributed::SingleProcessCollectives comm;
    auto params = cverl::distributed::module_parameter_views(module);
    cverl::distributed::broadcast_parameters_from_root(params, comm, 0, {0});

    require(torch::allclose(module.w, torch::arange(6, torch::kFloat32).reshape({2, 3})), "w unchanged");
    require(torch::allclose(module.b, torch::ones({3}, torch::kFloat32)), "b unchanged");

    CountingCollectives counting;
    cverl::distributed::broadcast_parameters_from_root(params, counting, 0, {0}, 1024 * 1024);
    require(counting.broadcast_calls == 0, "single-rank broadcast should skip payload broadcast");
    require(counting.barrier_calls == 0, "single-rank broadcast should skip final barrier");

    cverl::distributed::broadcast_parameters_from_root(params, counting, 0, {0, 1}, 1024 * 1024);
    require(counting.broadcast_calls == 1, "same dtype/device params should share one broadcast bucket");
    require(counting.barrier_calls == 1, "broadcast should keep one final barrier");

    counting.broadcast_calls = 0;
    counting.barrier_calls = 0;
    cverl::distributed::broadcast_parameters_from_root(params, counting, 0, {0, 1}, 16);
    require(counting.broadcast_calls > 1, "small broadcast bucket should split calls");
    require(counting.last_broadcast_numel == module.b.numel(),
            "single-parameter broadcast bucket should preserve payload numel");
    require(counting.barrier_calls == 1, "split broadcast should keep one final barrier");

    counting.broadcast_calls = 0;
    counting.barrier_calls = 0;
    cverl::distributed::broadcast_parameters_from_root(params, counting, 0, {0, 1}, 16, false);
    require(counting.broadcast_calls > 1, "barrier-free split broadcast still sends buckets");
    require(counting.barrier_calls == 0, "barrier-free broadcast should not call barrier");

    bool rejected = false;
    try {
      cverl::distributed::broadcast_parameters_from_root(params, comm, 1, {0});
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "invalid root rejected");

    std::cout << "weight sync tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_weight_sync failed: " << e.what() << "\n";
    return 1;
  }
}
