#include <algorithm>
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

std::vector<int64_t> group_ranks(int64_t n) {
  std::vector<int64_t> ranks;
  ranks.reserve(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    ranks.push_back(i);
  }
  return ranks;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      throw std::runtime_error("usage: qwen3_5_dp_tp_smoke MODEL_DIR --global-rank R --dp-size D --tp-size T");
    }
    std::string model_dir = argv[1];
    int64_t global_rank = arg_i64(argc, argv, "--global-rank", std::getenv("RANK") ? std::stoll(std::getenv("RANK")) : 0);
    int64_t dp_size = arg_i64(argc, argv, "--dp-size", 2);
    int64_t tp_size = arg_i64(argc, argv, "--tp-size", 2);
    int64_t device = arg_i64(argc, argv, "--device", std::getenv("LOCAL_RANK") ? std::stoll(std::getenv("LOCAL_RANK")) : global_rank);
    int64_t layers = arg_i64(argc, argv, "--layers", 4);
    std::string id_prefix = arg_str(argc, argv, "--id-prefix", "/tmp/cverl_qwen_dp_tp");

    int64_t dp_rank = global_rank / tp_size;
    int64_t tp_rank = global_rank % tp_size;
    if (dp_rank < 0 || dp_rank >= dp_size) {
      throw std::invalid_argument("global_rank outside dp/tp grid");
    }

    std::string tp_id_path = id_prefix + "_tp_" + std::to_string(dp_rank) + ".bin";
    std::string dp_id_path = id_prefix + "_dp_" + std::to_string(tp_rank) + ".bin";
    if (tp_rank == 0) {
      cverl::distributed::write_nccl_unique_id_file(tp_id_path, cverl::distributed::create_nccl_unique_id());
    }
    if (dp_rank == 0) {
      cverl::distributed::write_nccl_unique_id_file(dp_id_path, cverl::distributed::create_nccl_unique_id());
    }

    auto tp_id = cverl::distributed::read_nccl_unique_id_file(tp_id_path);
    auto dp_id = cverl::distributed::read_nccl_unique_id_file(dp_id_path);
    cverl::distributed::NcclCollectives tp_comm(tp_rank, tp_size, static_cast<int>(device), tp_id);
    cverl::distributed::NcclCollectives dp_comm(dp_rank, dp_size, static_cast<int>(device), dp_id);
    auto tp_ranks = group_ranks(tp_size);
    auto dp_ranks = group_ranks(dp_size);
    cverl::distributed::ParallelGroup tp_group{tp_rank, tp_size, tp_ranks, &tp_comm};

    bool qwen_ok = true;
    double max_abs = 0.0;
    double mean_abs = 0.0;
    int64_t qwen_cases = 0;
    {
      torch::NoGradGuard no_grad;
      cverl::HfModelLoader loader(model_dir);
      cverl::Qwen35TextModel model(std::move(loader));
      model.to(torch::Device(torch::kCUDA, static_cast<int>(device)));
      std::vector<torch::Tensor> cases{
          torch::tensor({1, 2}, torch::kLong).view({1, 2}),
          torch::tensor({10, 11, 12, 13}, torch::kLong).view({1, 4}),
      };
      qwen_cases = static_cast<int64_t>(cases.size());
      for (const auto& ids : cases) {
        auto dense = model.forward_hidden(ids, layers);
        auto tp = model.forward_hidden_tensor_parallel(ids, tp_group, layers);
        auto diff = (dense - tp).abs();
        qwen_ok = qwen_ok && torch::allclose(dense, tp, 1.0e-4, 1.0e-4);
        max_abs = std::max(max_abs, diff.max().item<double>());
        mean_abs = std::max(mean_abs, diff.mean().item<double>());
      }
    }

    auto param = torch::ones({2, 3}, torch::TensorOptions()
                                       .device(torch::Device(torch::kCUDA, static_cast<int>(device)))
                                       .dtype(torch::kFloat32)
                                       .requires_grad(true));
    auto loss = (param * static_cast<double>(dp_rank + 1)).sum();
    loss.backward();
    cverl::distributed::data_parallel_sync_gradients({param}, dp_comm, dp_ranks, true);
    auto expected_grad = torch::full({2, 3}, static_cast<float>(dp_size + 1) / 2.0f,
                                     torch::TensorOptions().device(param.device()).dtype(torch::kFloat32));
    bool dp_ok = torch::allclose(param.grad(), expected_grad, 1.0e-5, 1.0e-6);

    tp_comm.barrier();
    dp_comm.barrier();
    if (global_rank == 0) {
      for (int64_t d = 0; d < dp_size; ++d) {
        std::filesystem::remove(id_prefix + "_tp_" + std::to_string(d) + ".bin");
      }
      for (int64_t t = 0; t < tp_size; ++t) {
        std::filesystem::remove(id_prefix + "_dp_" + std::to_string(t) + ".bin");
      }
      std::cout << "qwen3.5 dp/tp smoke dp=" << dp_size << " tp=" << tp_size << " layers=" << layers << "\n";
      std::cout << "qwen_cases=" << qwen_cases << "\n";
      std::cout << "qwen_tp_max_abs=" << max_abs << "\n";
      std::cout << "qwen_tp_mean_abs=" << mean_abs << "\n";
      std::cout << "qwen_tp_allclose=" << (qwen_ok ? "true" : "false") << "\n";
      std::cout << "dp_grad_allclose=" << (dp_ok ? "true" : "false") << "\n";
    }
    return qwen_ok && dp_ok ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "qwen3_5_dp_tp_smoke failed: " << e.what() << "\n";
    return 1;
  }
}
