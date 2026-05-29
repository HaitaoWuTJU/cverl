#include "cverl/distributed/nccl_collectives.h"
#include "cverl/distributed/parallel_ops.h"

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

    auto x = torch::full({4}, static_cast<float>(rank + 1),
                         torch::TensorOptions().device(torch::kCUDA, static_cast<int>(device)).dtype(torch::kFloat32));
    auto reduced = comm.all_reduce(x, cverl::distributed::ReduceOp::Sum, group);
    auto expected_sum = torch::full({4}, static_cast<float>(world * (world + 1) / 2),
                                    torch::TensorOptions().device(torch::kCUDA, static_cast<int>(device)).dtype(torch::kFloat32));
    require_allclose(reduced, expected_sum, "NCCL all_reduce sum mismatch");

    auto mean = comm.all_reduce(x, cverl::distributed::ReduceOp::Mean, group);
    auto expected_mean = expected_sum / static_cast<double>(world);
    require_allclose(mean, expected_mean, "NCCL all_reduce mean mismatch");

    auto gathered = comm.all_gather(torch::full({1, 2}, static_cast<float>(rank),
                                                torch::TensorOptions().device(torch::kCUDA, static_cast<int>(device)).dtype(torch::kFloat32)),
                                    group, 0);
    auto expected_gather = torch::arange(0, world, torch::TensorOptions().device(torch::kCUDA, static_cast<int>(device)).dtype(torch::kFloat32))
                               .view({world, 1})
                               .repeat({1, 2});
    require_allclose(gathered, expected_gather, "NCCL all_gather mismatch");

    auto scatter_in = torch::full({world, 2}, static_cast<float>(rank + 1),
                                  torch::TensorOptions().device(torch::kCUDA, static_cast<int>(device)).dtype(torch::kFloat32));
    auto scattered = comm.reduce_scatter(scatter_in, cverl::distributed::ReduceOp::Sum, group, 0);
    auto expected_scatter = torch::full({1, 2}, static_cast<float>(world * (world + 1) / 2),
                                        torch::TensorOptions().device(torch::kCUDA, static_cast<int>(device)).dtype(torch::kFloat32));
    require_allclose(scattered, expected_scatter, "NCCL reduce_scatter mismatch");

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
