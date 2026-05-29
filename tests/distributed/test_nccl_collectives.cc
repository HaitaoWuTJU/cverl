#include "cverl/distributed/nccl_collectives.h"

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
