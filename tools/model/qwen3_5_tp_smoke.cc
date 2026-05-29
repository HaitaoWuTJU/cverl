#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "cverl/distributed/nccl_collectives.h"
#include "cverl/distributed/parallel_ops.h"
#include "cverl/model/hf_model_loader.h"
#include "cverl/model/qwen3_5_text.h"

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

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      throw std::runtime_error("usage: qwen3_5_tp_smoke MODEL_DIR [--rank R] [--world-size N] [--device D] [--layers N]");
    }
    std::string model_dir = argv[1];
    int64_t rank = arg_i64(argc, argv, "--rank", std::getenv("RANK") ? std::stoll(std::getenv("RANK")) : 0);
    int64_t world = arg_i64(argc, argv, "--world-size", std::getenv("WORLD_SIZE") ? std::stoll(std::getenv("WORLD_SIZE")) : 1);
    int64_t device = arg_i64(argc, argv, "--device", std::getenv("LOCAL_RANK") ? std::stoll(std::getenv("LOCAL_RANK")) : rank);
    int64_t layers = arg_i64(argc, argv, "--layers", 4);
    std::string id_path = arg_str(argc, argv, "--id-file", "/tmp/cverl_qwen_tp_nccl_unique_id.bin");

    torch::NoGradGuard no_grad;
    if (rank == 0) {
      cverl::distributed::write_nccl_unique_id_file(id_path, cverl::distributed::create_nccl_unique_id());
    }
    auto id = cverl::distributed::read_nccl_unique_id_file(id_path);
    cverl::distributed::NcclCollectives comm(rank, world, static_cast<int>(device), id);
    auto ranks = full_group(world);
    cverl::distributed::ParallelGroup tp_group{rank, world, ranks, &comm};

    cverl::HfModelLoader loader(model_dir);
    cverl::Qwen35TextModel model(std::move(loader));
    model.to(torch::Device(torch::kCUDA, static_cast<int>(device)));

    auto ids = torch::tensor({1, 2}, torch::kLong).view({1, 2});
    auto dense = model.forward_hidden(ids, layers);
    auto tp = model.forward_hidden_tensor_parallel(ids, tp_group, layers);
    auto diff = (dense - tp).abs();
    double max_abs = diff.max().item<double>();
    double mean_abs = diff.mean().item<double>();
    bool ok = torch::allclose(dense, tp, 1.0e-4, 1.0e-4);
    comm.barrier();
    if (rank == 0) {
      std::filesystem::remove(id_path);
      std::cout << "qwen3.5 tp smoke layers=" << layers << " world=" << world << "\n";
      std::cout << "max_abs=" << max_abs << "\n";
      std::cout << "mean_abs=" << mean_abs << "\n";
      std::cout << "allclose=" << (ok ? "true" : "false") << "\n";
    }
    return ok ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "qwen3_5_tp_smoke failed: " << e.what() << "\n";
    return 1;
  }
}
