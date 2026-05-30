#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "cverl/distributed/nccl_collectives.h"
#include "cverl/distributed/parallel_ops.h"
#include "cverl/distributed/topology.h"
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

std::vector<int64_t> full_group(int64_t size) {
  std::vector<int64_t> ranks;
  ranks.reserve(static_cast<size_t>(size));
  for (int64_t i = 0; i < size; ++i) {
    ranks.push_back(i);
  }
  return ranks;
}

torch::ScalarType parse_dtype(const std::string& dtype) {
  if (dtype == "float32" || dtype == "fp32") {
    return torch::kFloat32;
  }
  if (dtype == "bfloat16" || dtype == "bf16") {
    return torch::kBFloat16;
  }
  if (dtype == "float16" || dtype == "fp16") {
    return torch::kFloat16;
  }
  throw std::invalid_argument("unsupported dtype");
}

double grad_norm_sum(const std::vector<torch::Tensor>& params) {
  double out = 0.0;
  for (const auto& p : params) {
    if (p.grad().defined()) {
      out += p.grad().detach().to(torch::kFloat32).norm().item<double>();
    }
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      throw std::runtime_error(
          "usage: qwen3_5_pp_tp_train_smoke MODEL_DIR --global-rank R --device D --layers N");
    }

    const std::string model_dir = argv[1];
    const int64_t global_rank =
        arg_i64(argc, argv, "--global-rank", std::getenv("RANK") ? std::stoll(std::getenv("RANK")) : 0);
    const int64_t pp_size = arg_i64(argc, argv, "--pp-size", 2);
    const int64_t tp_size = arg_i64(argc, argv, "--tp-size", 2);
    const int64_t world_size = pp_size * tp_size;
    const int64_t device =
        arg_i64(argc, argv, "--device", std::getenv("LOCAL_RANK") ? std::stoll(std::getenv("LOCAL_RANK")) : global_rank);
    const int64_t layers = arg_i64(argc, argv, "--layers", 2);
    const int64_t seq_len = arg_i64(argc, argv, "--seq-len", 4);
    const auto dtype = parse_dtype(arg_str(argc, argv, "--dtype", "bfloat16"));
    const std::string id_prefix = arg_str(argc, argv, "--id-prefix", "/tmp/cverl_qwen_pp_tp_train");

    cverl::distributed::ParallelDims dims;
    dims.data_parallel = 1;
    dims.pipeline_parallel = pp_size;
    dims.context_parallel = 1;
    dims.tensor_parallel = tp_size;
    dims.micro_batches = pp_size;
    cverl::distributed::ClusterSpec spec;
    spec.world_size = world_size;
    spec.rank = global_rank;
    spec.local_world_size = world_size;
    spec.local_rank = device;
    spec.gpus_per_node = world_size;
    spec.parallel = dims;
    cverl::distributed::Topology topology(spec);
    auto info = topology.local_rank_info();
    auto range = topology.pipeline_layer_range(layers, info.pipeline_rank);

    const std::string full_id_path = id_prefix + "_full.bin";
    const std::string tp_id_path = id_prefix + "_tp_pp" + std::to_string(info.pipeline_rank) + ".bin";
    if (global_rank == 0) {
      cverl::distributed::write_nccl_unique_id_file(full_id_path, cverl::distributed::create_nccl_unique_id());
    }
    if (info.tensor_rank == 0) {
      cverl::distributed::write_nccl_unique_id_file(tp_id_path, cverl::distributed::create_nccl_unique_id());
    }

    auto full_id = cverl::distributed::read_nccl_unique_id_file(full_id_path);
    auto tp_id = cverl::distributed::read_nccl_unique_id_file(tp_id_path);
    cverl::distributed::NcclCollectives full_comm(global_rank, world_size, static_cast<int>(device), full_id);
    cverl::distributed::NcclCollectives tp_comm(info.tensor_rank, tp_size, static_cast<int>(device), tp_id);
    cverl::distributed::ParallelGroup tp_group{info.tensor_rank, tp_size, full_group(tp_size), &tp_comm};

    cverl::HfModelLoader loader(model_dir);
    cverl::Qwen35TextModel model(std::move(loader));
    model.to(torch::Device(torch::kCUDA, static_cast<int>(device)));
    std::vector<torch::Tensor> params;
    for (const auto& name : model.required_weight_names(layers)) {
      auto tensor = model.loader()
                        .load_tensor(name)
                        .to(torch::TensorOptions().device(torch::kCUDA, static_cast<int>(device)).dtype(dtype))
                        .contiguous();
      tensor.set_requires_grad(true);
      model.set_weight_override(name, tensor);
      params.push_back(tensor);
    }

    const auto ids = torch::arange(1, seq_len + 1, torch::kLong)
                         .view({1, seq_len})
                         .to(torch::Device(torch::kCUDA, static_cast<int>(device)));
    const std::vector<int64_t> hidden_shape{1, seq_len, model.config().hidden_size};
    torch::Tensor activation;

    if (info.pipeline_rank == 0) {
      auto hidden = model.token_embeddings(ids);
      activation = model.forward_hidden_range_tensor_parallel(hidden, range.begin, range.end, tp_group, false);
      full_comm.send(activation.contiguous(), global_rank + tp_size);
      auto grad = full_comm.recv_like(activation.contiguous(), global_rank + tp_size);
      activation.backward(grad);
    } else if (info.pipeline_rank + 1 == pp_size) {
      auto like = torch::empty(hidden_shape, torch::TensorOptions().device(torch::kCUDA, static_cast<int>(device)).dtype(dtype));
      auto input = full_comm.recv_like(like, global_rank - tp_size).detach().set_requires_grad(true);
      auto hidden = model.forward_hidden_range_tensor_parallel(input, range.begin, range.end, tp_group, true);
      auto loss = hidden.to(torch::kFloat32).square().mean();
      loss.backward();
      full_comm.send(input.grad().contiguous(), global_rank - tp_size);
    } else {
      throw std::runtime_error("qwen3_5_pp_tp_train_smoke currently expects pp-size=2");
    }

    const double local_grad_norm = grad_norm_sum(params);
    auto grad_tensor = torch::tensor({local_grad_norm}, torch::TensorOptions().device(torch::kCUDA, static_cast<int>(device)).dtype(torch::kFloat32));
    auto grad_norms = full_comm.all_gather(grad_tensor.contiguous(), full_group(world_size), 0).cpu();
    full_comm.barrier();

    if (global_rank == 0) {
      std::filesystem::remove(full_id_path);
      for (int64_t pp = 0; pp < pp_size; ++pp) {
        std::filesystem::remove(id_prefix + "_tp_pp" + std::to_string(pp) + ".bin");
      }
      std::cout << "qwen3.5 pp/tp train smoke pp=" << pp_size << " tp=" << tp_size << " layers=" << layers
                << " seq_len=" << seq_len << "\n";
      bool all_have_grad = true;
      for (int64_t i = 0; i < grad_norms.numel(); ++i) {
        const double value = grad_norms[i].item<double>();
        std::cout << "rank" << i << "_grad_norm_sum=" << value << "\n";
        all_have_grad = all_have_grad && value > 0.0;
      }
      std::cout << "all_ranks_have_grad=" << (all_have_grad ? "true" : "false") << "\n";
      return all_have_grad ? 0 : 2;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "qwen3_5_pp_tp_train_smoke failed: " << e.what() << "\n";
    return 1;
  }
}
