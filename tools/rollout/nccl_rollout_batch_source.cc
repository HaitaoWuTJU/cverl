#include "cverl/distributed/nccl_collectives.h"
#include "cverl/rollout/nccl_gpu_batch_transport.h"

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
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

std::vector<int64_t> parse_csv_i64(const std::string& text) {
  std::vector<int64_t> out;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      out.push_back(std::stoll(item));
    }
  }
  return out;
}

cverl::rollout::GpuRolloutBatch make_batch(int64_t step,
                                           int64_t batch_size,
                                           int64_t prompt_len,
                                           int64_t response_len,
                                           int64_t weight_version,
                                           torch::Device device) {
  if (batch_size <= 0 || prompt_len <= 0 || response_len <= 0) {
    throw std::invalid_argument("batch, prompt_len, and response_len must be positive");
  }
  cverl::rollout::GpuRolloutBatch batch;
  batch.descriptor.step = step;
  batch.descriptor.weight_version = weight_version;
  batch.prompt_ids =
      torch::arange(1, batch_size * prompt_len + 1, torch::TensorOptions().device(device).dtype(torch::kInt64))
          .view({batch_size, prompt_len})
          .contiguous();
  batch.response_ids =
      torch::arange(100, 100 + batch_size * response_len, torch::TensorOptions().device(device).dtype(torch::kInt64))
          .view({batch_size, response_len})
          .contiguous();
  batch.response_mask =
      torch::ones({batch_size, response_len}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
  auto row = torch::arange(0, batch_size, torch::TensorOptions().device(device).dtype(torch::kFloat32))
                 .view({batch_size, 1});
  batch.advantages = torch::where(
      row.remainder(2).eq(0),
      torch::ones({batch_size, response_len}, torch::TensorOptions().device(device).dtype(torch::kFloat32)),
      -torch::ones({batch_size, response_len}, torch::TensorOptions().device(device).dtype(torch::kFloat32)));
  batch.old_log_probs =
      torch::zeros({batch_size, response_len}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
  batch.rewards = row.remainder(2).eq(0).to(torch::kFloat32).view({batch_size}).contiguous();
  batch.group_ids = torch::arange(0, batch_size, torch::TensorOptions().device(device).dtype(torch::kInt64))
                        .floor_divide(2)
                        .contiguous();
  return batch;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const int64_t rank = arg_i64(argc, argv, "--rank", std::getenv("RANK") ? std::stoll(std::getenv("RANK")) : 0);
    const int64_t world =
        arg_i64(argc, argv, "--world-size", std::getenv("WORLD_SIZE") ? std::stoll(std::getenv("WORLD_SIZE")) : 1);
    const int64_t device =
        arg_i64(argc, argv, "--device", std::getenv("LOCAL_RANK") ? std::stoll(std::getenv("LOCAL_RANK")) : 0);
    const std::string id_file = arg_str(argc, argv, "--id-file", "/tmp/cverl_rollout_data_nccl.bin");
    const auto targets = parse_csv_i64(arg_str(argc, argv, "--target-ranks", ""));
    const int64_t steps = arg_i64(argc, argv, "--steps", 1);
    const int64_t batch_size = arg_i64(argc, argv, "--batch", 4);
    const int64_t prompt_len = arg_i64(argc, argv, "--prompt-len", 32);
    const int64_t response_len = arg_i64(argc, argv, "--response-len", 8);
    const int64_t weight_version = arg_i64(argc, argv, "--weight-version", 0);

    if (rank == 0) {
      cverl::distributed::write_nccl_unique_id_file(id_file, cverl::distributed::create_nccl_unique_id());
    }
    auto id = cverl::distributed::read_nccl_unique_id_file(id_file);
    cverl::distributed::NcclCollectives comm(rank, world, static_cast<int>(device), id);
    cverl::rollout::NCCLGpuBatchSender sender(comm);
    torch::Device dev(torch::kCUDA, static_cast<int>(device));

    for (int64_t step = 1; step <= steps; ++step) {
      auto batch = make_batch(step, batch_size, prompt_len, response_len, weight_version + step, dev);
      for (int64_t peer : targets) {
        if (peer < 0 || peer >= world || peer == rank) {
          throw std::invalid_argument("invalid target rank");
        }
        sender.send(batch, peer);
      }
      comm.barrier();
      std::cout << "rollout source step " << step << " sent batch=" << batch_size
                << " prompt_len=" << prompt_len << " response_len=" << response_len
                << " targets=" << targets.size() << "\n";
    }

    if (rank == 0) {
      std::filesystem::remove(id_file);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "nccl_rollout_batch_source failed: " << e.what() << "\n";
    return 1;
  }
}
