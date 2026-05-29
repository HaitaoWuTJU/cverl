#include "cverl/distributed/parallel_ops.h"

#include <iostream>
#include <stdexcept>
#include <vector>

#include <torch/torch.h>

namespace {

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

torch::Tensor dense_linear(const torch::Tensor& input, const torch::Tensor& weight) {
  return torch::matmul(input, weight.transpose(0, 1));
}

void test_shard_dim() {
  auto x = torch::arange(24, torch::kFloat32).reshape({2, 12});
  require_allclose(cverl::distributed::shard_dim(x, 1, 1, 3), x.narrow(1, 4, 4), "shard dim");
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

void test_dp_sync_single_process() {
  cverl::distributed::SingleProcessCollectives collectives;
  auto p = torch::randn({2, 3}, torch::requires_grad());
  auto loss = (p * p).sum();
  loss.backward();
  auto before = p.grad().clone();
  cverl::distributed::data_parallel_sync_gradients({p}, collectives, {0}, true);
  require_allclose(p.grad(), before, "single process dp sync");
}

}  // namespace

int main() {
  try {
    test_shard_dim();
    test_tp_linear_manual_reduce();
    test_tp_mlp_manual_reduce();
    test_dp_sync_single_process();
  } catch (const std::exception& e) {
    std::cerr << "test_parallel_ops failed: " << e.what() << "\n";
    return 1;
  }
  std::cout << "distributed parallel ops tests passed\n";
  return 0;
}
