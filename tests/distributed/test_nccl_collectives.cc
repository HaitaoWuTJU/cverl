#include "cverl/distributed/nccl_collectives.h"
#include "cverl/distributed/parallel_ops.h"
#include "cverl/distributed/weight_sync.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <torch/torch.h>

namespace {

int64_t arg_i64(int argc, char** argv, const std::string& flag, int64_t fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == flag) {
      return std::stoll(argv[i + 1]);
    }
  }
  return fallback;
}

std::string arg_str(int argc, char** argv, const std::string& flag, const std::string& fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == flag) {
      return argv[i + 1];
    }
  }
  return fallback;
}

std::vector<int64_t> full_group(int64_t world_size) {
  std::vector<int64_t> group;
  group.reserve(static_cast<size_t>(world_size));
  for (int64_t i = 0; i < world_size; ++i) {
    group.push_back(i);
  }
  return group;
}

void require_allclose(const torch::Tensor& actual, const torch::Tensor& expected, const char* msg) {
  if (!torch::allclose(actual.cpu(), expected.cpu(), 1.0e-5, 1.0e-6)) {
    std::cerr << msg << "\nactual=" << actual.cpu() << "\nexpected=" << expected.cpu() << "\n";
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

}  // namespace

int main(int argc, char** argv) {
  try {
    int64_t rank = arg_i64(argc, argv, "--rank", std::getenv("RANK") ? std::stoll(std::getenv("RANK")) : 0);
    int64_t world = arg_i64(argc, argv, "--world-size", std::getenv("WORLD_SIZE") ? std::stoll(std::getenv("WORLD_SIZE")) : 1);
    int64_t device = arg_i64(argc, argv, "--device", std::getenv("LOCAL_RANK") ? std::stoll(std::getenv("LOCAL_RANK")) : rank);
    std::string id_path = arg_str(argc, argv, "--id-file", "/tmp/cverl_nccl_unique_id.bin");

    if (rank == 0) {
      cverl::distributed::write_nccl_unique_id_file(id_path, cverl::distributed::create_nccl_unique_id());
    }
    auto id = cverl::distributed::read_nccl_unique_id_file(id_path);
    cverl::distributed::NcclCollectives comm(rank, world, static_cast<int>(device), id);
    auto group = full_group(world);

    auto bcast_in = torch::full({4}, rank == 0 ? 7.0f : -1.0f,
                                torch::TensorOptions().device(torch::kCUDA, static_cast<int>(device)).dtype(torch::kFloat32));
    auto bcast = comm.broadcast(bcast_in.contiguous(), 0, group);
    auto expected_bcast = torch::full({4}, 7.0f,
                                      torch::TensorOptions().device(torch::kCUDA, static_cast<int>(device)).dtype(torch::kFloat32));
    require_allclose(bcast, expected_bcast, "NCCL broadcast mismatch");

    auto synced_param = torch::full({2, 3}, rank == 0 ? 3.0f : -5.0f,
                                    torch::TensorOptions().device(torch::kCUDA, static_cast<int>(device)).dtype(torch::kFloat32));
    cverl::distributed::broadcast_parameters_from_root(
        {cverl::distributed::ParameterView{"synced_param", synced_param}}, comm, 0, group);
    auto expected_param = torch::full({2, 3}, 3.0f,
                                      torch::TensorOptions().device(torch::kCUDA, static_cast<int>(device)).dtype(torch::kFloat32));
    require_allclose(synced_param, expected_param, "NCCL parameter broadcast mismatch");

    auto x = torch::full({4}, static_cast<float>(rank + 1),
                         torch::TensorOptions().device(torch::kCUDA, static_cast<int>(device)).dtype(torch::kFloat32));
    auto reduced = comm.all_reduce(x, cverl::distributed::ReduceOp::Sum, group);
    auto expected_sum = torch::full({4}, static_cast<float>(world * (world + 1) / 2),
                                    torch::TensorOptions().device(torch::kCUDA, static_cast<int>(device)).dtype(torch::kFloat32));
    require_allclose(reduced, expected_sum, "NCCL all_reduce sum mismatch");

    auto mean = comm.all_reduce(x, cverl::distributed::ReduceOp::Mean, group);
    auto expected_mean = expected_sum / static_cast<double>(world);
    require_allclose(mean, expected_mean, "NCCL all_reduce mean mismatch");

    auto maxed = comm.all_reduce(x, cverl::distributed::ReduceOp::Max, group);
    auto expected_max = torch::full({4}, static_cast<float>(world),
                                    torch::TensorOptions().device(torch::kCUDA, static_cast<int>(device)).dtype(torch::kFloat32));
    require_allclose(maxed, expected_max, "NCCL all_reduce max mismatch");

    auto matrix = torch::arange(8, torch::TensorOptions().device(torch::kCUDA, static_cast<int>(device)).dtype(torch::kFloat32))
                      .reshape({2, 4});
    require_throws(
        [&]() { (void)comm.all_reduce(matrix.transpose(0, 1), cverl::distributed::ReduceOp::Sum, group); },
        "NCCL all_reduce should reject non-contiguous tensors");

    auto gathered = comm.all_gather(torch::full({1, 2}, static_cast<float>(rank),
                                                torch::TensorOptions().device(torch::kCUDA, static_cast<int>(device)).dtype(torch::kFloat32)),
                                    group, 0);
    auto expected_gather = torch::arange(0, world, torch::TensorOptions().device(torch::kCUDA, static_cast<int>(device)).dtype(torch::kFloat32))
                               .view({world, 1})
                               .repeat({1, 2});
    require_allclose(gathered, expected_gather, "NCCL all_gather mismatch");
    require_throws([&]() { (void)comm.all_gather(gathered.contiguous(), group, 1); },
                   "NCCL all_gather should reject dim != 0");

    auto scatter_in = torch::full({world, 2}, static_cast<float>(rank + 1),
                                  torch::TensorOptions().device(torch::kCUDA, static_cast<int>(device)).dtype(torch::kFloat32));
    auto scattered = comm.reduce_scatter(scatter_in, cverl::distributed::ReduceOp::Sum, group, 0);
    auto expected_scatter = torch::full({1, 2}, static_cast<float>(world * (world + 1) / 2),
                                        torch::TensorOptions().device(torch::kCUDA, static_cast<int>(device)).dtype(torch::kFloat32));
    require_allclose(scattered, expected_scatter, "NCCL reduce_scatter mismatch");

    auto scattered_mean = comm.reduce_scatter(scatter_in, cverl::distributed::ReduceOp::Mean, group, 0);
    auto expected_scatter_mean = expected_scatter / static_cast<double>(world);
    require_allclose(scattered_mean, expected_scatter_mean, "NCCL reduce_scatter mean mismatch");
    require_throws([&]() { (void)comm.reduce_scatter(scatter_in.contiguous(), cverl::distributed::ReduceOp::Sum, group, 1); },
                   "NCCL reduce_scatter should reject dim != 0");

    torch::manual_seed(123);
    auto cpu_x = torch::randn({2, 3, 8});
    auto cpu_gate = torch::randn({16, 8});
    auto cpu_up = torch::randn({16, 8});
    auto cpu_down = torch::randn({8, 16});
    auto mlp_x = cpu_x.to(torch::Device(torch::kCUDA, static_cast<int>(device)));
    auto gate = cpu_gate.to(mlp_x.device());
    auto up = cpu_up.to(mlp_x.device());
    auto down = cpu_down.to(mlp_x.device());
    cverl::distributed::ParallelGroup tp_group{rank, world, group, &comm};
    auto tp_mlp = cverl::distributed::tensor_parallel_mlp_swiglu(mlp_x, gate, up, down, tp_group);
    auto dense_gate = torch::matmul(mlp_x, gate.transpose(0, 1));
    auto dense_up = torch::matmul(mlp_x, up.transpose(0, 1));
    auto dense_mlp = torch::matmul((torch::silu(dense_gate) * dense_up), down.transpose(0, 1));
    require_allclose(tp_mlp, dense_mlp, "NCCL tensor-parallel MLP mismatch");

    auto param = torch::ones({2, 3}, torch::TensorOptions().device(mlp_x.device()).dtype(torch::kFloat32).requires_grad(true));
    auto loss = (param * static_cast<double>(rank + 1)).sum();
    loss.backward();
    cverl::distributed::data_parallel_sync_gradients({param}, comm, group, true);
    auto expected_grad = torch::full({2, 3}, static_cast<float>(world + 1) / 2.0f,
                                     torch::TensorOptions().device(mlp_x.device()).dtype(torch::kFloat32));
    require_allclose(param.grad(), expected_grad, "NCCL data-parallel gradient sync mismatch");

    if (world >= 2) {
      auto pipe_like = torch::empty({2, 2}, torch::TensorOptions().device(torch::kCUDA, static_cast<int>(device)).dtype(torch::kFloat32));
      if (rank == 0) {
        auto payload = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}, torch::TensorOptions().device(pipe_like.device())).reshape({2, 2});
        comm.send(payload.contiguous(), 1);
      } else if (rank == 1) {
        auto received = comm.recv_like(pipe_like, 0);
        auto expected = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}, torch::TensorOptions().device(pipe_like.device())).reshape({2, 2});
        require_allclose(received, expected, "NCCL pipeline send/recv mismatch");
      }
    }

    if (world >= 2) {
      auto shard0 = torch::zeros({2, 2}, torch::TensorOptions().device(torch::kCUDA, static_cast<int>(device)).dtype(torch::kFloat32));
      auto shard1 = torch::zeros({3}, torch::TensorOptions().device(torch::kCUDA, static_cast<int>(device)).dtype(torch::kFloat32));
      if (rank == 0) {
        shard0.copy_(torch::arange(1, 5, torch::TensorOptions().device(shard0.device()).dtype(torch::kFloat32)).reshape({2, 2}));
        shard1.copy_(torch::tensor({11.0f, 12.0f, 13.0f}, torch::TensorOptions().device(shard1.device())));
        cverl::distributed::send_parameter_shards(
            {cverl::distributed::ParameterView{"tp_linear.weight.shard0", shard0},
             cverl::distributed::ParameterView{"tp_linear.bias.shard0", shard1}},
            comm,
            1);
      } else if (rank == 1) {
        cverl::distributed::recv_parameter_shards(
            {cverl::distributed::ParameterView{"tp_linear.weight.shard0", shard0},
             cverl::distributed::ParameterView{"tp_linear.bias.shard0", shard1}},
            comm,
            0);
        auto expected_shard0 =
            torch::arange(1, 5, torch::TensorOptions().device(shard0.device()).dtype(torch::kFloat32)).reshape({2, 2});
        auto expected_shard1 = torch::tensor({11.0f, 12.0f, 13.0f}, torch::TensorOptions().device(shard1.device()));
        require_allclose(shard0, expected_shard0, "NCCL shard-wise parameter sync weight mismatch");
        require_allclose(shard1, expected_shard1, "NCCL shard-wise parameter sync bias mismatch");
      }
    }

    comm.barrier();
    if (rank == 0) {
      std::filesystem::remove(id_path);
    }
    std::cout << "nccl collectives rank " << rank << " passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_nccl_collectives failed: " << e.what() << "\n";
    return 1;
  }
}
