#include "cverl/distributed/nccl_collectives.h"
#include "cverl/rollout/nccl_gpu_batch_transport.h"

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

void require(bool cond, const char* msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

void require_allclose(const torch::Tensor& got, const torch::Tensor& expected, const char* msg) {
  if (!torch::allclose(got.cpu(), expected.cpu(), 1e-5, 1e-6)) {
    std::cerr << msg << "\ngot=" << got.cpu() << "\nexpected=" << expected.cpu() << "\n";
    throw std::runtime_error(msg);
  }
}

std::vector<int64_t> full_group(int64_t world) {
  std::vector<int64_t> out;
  for (int64_t i = 0; i < world; ++i) {
    out.push_back(i);
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int64_t rank = arg_i64(argc, argv, "--rank", std::getenv("RANK") ? std::stoll(std::getenv("RANK")) : 0);
    int64_t world = arg_i64(argc, argv, "--world-size", std::getenv("WORLD_SIZE") ? std::stoll(std::getenv("WORLD_SIZE")) : 2);
    int64_t device = arg_i64(argc, argv, "--device", std::getenv("LOCAL_RANK") ? std::stoll(std::getenv("LOCAL_RANK")) : rank);
    std::string id_path = arg_str(argc, argv, "--id-file", "/tmp/cverl_nccl_gpu_batch_id.bin");
    require(world >= 2, "NCCL GPU batch test requires at least 2 ranks");

    if (rank == 0) {
      cverl::distributed::write_nccl_unique_id_file(id_path, cverl::distributed::create_nccl_unique_id());
    }
    auto id = cverl::distributed::read_nccl_unique_id_file(id_path);
    cverl::distributed::NcclCollectives comm(rank, world, static_cast<int>(device), id);
    torch::Device dev(torch::kCUDA, static_cast<int>(device));

    if (rank == 0) {
      cverl::rollout::GpuRolloutBatch batch;
      batch.descriptor.step = 17;
      batch.descriptor.weight_version = 23;
      batch.prompt_ids = torch::arange(0, 12, torch::TensorOptions().device(dev).dtype(torch::kInt64)).view({3, 4}).contiguous();
      batch.response_ids = torch::arange(100, 106, torch::TensorOptions().device(dev).dtype(torch::kInt64)).view({3, 2}).contiguous();
      batch.response_mask = torch::ones({3, 2}, torch::TensorOptions().device(dev).dtype(torch::kFloat32));
      batch.advantages = torch::tensor({1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f}, torch::TensorOptions().device(dev)).view({3, 2}).contiguous();
      batch.old_log_probs = torch::full({3, 2}, -3.0f, torch::TensorOptions().device(dev).dtype(torch::kFloat32));
      batch.ref_log_probs = torch::full({3, 2}, -3.5f, torch::TensorOptions().device(dev).dtype(torch::kFloat32));
      batch.rewards = torch::tensor({1.0f, 0.0f, 1.0f}, torch::TensorOptions().device(dev));
      batch.group_ids = torch::tensor({0, 0, 1}, torch::TensorOptions().device(dev).dtype(torch::kInt64));
      cverl::rollout::NCCLGpuBatchSender sender(comm);
      sender.send(batch, 1);
    } else if (rank == 1) {
      cverl::rollout::NCCLGpuBatchReceiver receiver(comm, static_cast<int>(device));
      auto batch = receiver.recv(0);
      require(batch.descriptor.step == 17, "step mismatch");
      require(batch.descriptor.weight_version == 23, "weight version mismatch");
      require(batch.descriptor.batch == 3, "batch mismatch");
      require(batch.descriptor.prompt_len == 4, "prompt len mismatch");
      require(batch.descriptor.response_len == 2, "response len mismatch");
      require_allclose(batch.prompt_ids.to(torch::kFloat32),
                       torch::arange(0, 12, torch::TensorOptions().device(dev).dtype(torch::kFloat32)).view({3, 4}),
                       "prompt ids mismatch");
      require_allclose(batch.response_ids.to(torch::kFloat32),
                       torch::arange(100, 106, torch::TensorOptions().device(dev).dtype(torch::kFloat32)).view({3, 2}),
                       "response ids mismatch");
      require_allclose(batch.advantages,
                       torch::tensor({1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f}, torch::TensorOptions().device(dev)).view({3, 2}),
                       "advantages mismatch");
      require_allclose(batch.old_log_probs,
                       torch::full({3, 2}, -3.0f, torch::TensorOptions().device(dev).dtype(torch::kFloat32)),
                       "old log probs mismatch");
      require_allclose(batch.ref_log_probs,
                       torch::full({3, 2}, -3.5f, torch::TensorOptions().device(dev).dtype(torch::kFloat32)),
                       "ref log probs mismatch");
      cverl::rollout::TrainerIngressQueue queue(1);
      require(queue.push(std::move(batch)), "queue push failed");
      require(!queue.push(cverl::rollout::GpuRolloutBatch{}), "queue capacity failed");
      auto popped = queue.pop();
      require(popped.has_value(), "queue pop failed");
      require(queue.size() == 0, "queue size mismatch");
    }

    comm.barrier();
    if (rank == 0) {
      std::filesystem::remove(id_path);
    }
    std::cout << "nccl gpu batch transport rank " << rank << " passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_nccl_gpu_batch_transport failed: " << e.what() << "\n";
    return 1;
  }
}
