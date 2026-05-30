#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "cverl/distributed/nccl_collectives.h"
#include "cverl/distributed/parallel_ops.h"
#include "cverl/distributed/pipeline.h"
#include "cverl/distributed/topology.h"
#include "cverl/model/hf_model_loader.h"
#include "cverl/model/qwen3_5_text.h"
#include "cverl/torch/core_algos_torch.h"
#include "cverl/torch/tiny_causal_policy.h"

namespace {

int64_t arg_i64(int argc, char** argv, const std::string& flag, int64_t fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == flag) {
      return std::stoll(argv[i + 1]);
    }
  }
  return fallback;
}

double arg_f64(int argc, char** argv, const std::string& flag, double fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == flag) {
      return std::stod(argv[i + 1]);
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

double param_delta_sum(const std::vector<torch::Tensor>& before, const std::vector<torch::Tensor>& after) {
  double out = 0.0;
  for (size_t i = 0; i < before.size(); ++i) {
    out += (after[i].detach().to(torch::kFloat32) - before[i].to(torch::kFloat32)).abs().sum().item<double>();
  }
  return out;
}

std::vector<torch::Tensor> clone_params(const std::vector<torch::Tensor>& params) {
  std::vector<torch::Tensor> out;
  out.reserve(params.size());
  for (const auto& p : params) {
    out.push_back(p.detach().clone());
  }
  return out;
}

torch::Tensor make_token_ids(int64_t seq_len, int64_t micro_batch, int64_t step, const cverl::Qwen35TextConfig& config, torch::Device device) {
  auto ids = torch::arange(1, seq_len + 1, torch::TensorOptions().dtype(torch::kLong).device(device)).view({1, seq_len});
  const int64_t offset = step + micro_batch;
  return (ids + offset).remainder(config.vocab_size - 1).clamp_min(1);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      throw std::runtime_error(
          "usage: qwen3_5_pp_tp_ppo_trainer MODEL_DIR --global-rank R --device D --layers N");
    }

    const std::string model_dir = argv[1];
    const int64_t global_rank =
        arg_i64(argc, argv, "--global-rank", std::getenv("RANK") ? std::stoll(std::getenv("RANK")) : 0);
    const int64_t pp_size = arg_i64(argc, argv, "--pp-size", 2);
    const int64_t tp_size = arg_i64(argc, argv, "--tp-size", 2);
    const int64_t world_size = pp_size * tp_size;
    const int64_t device_idx =
        arg_i64(argc, argv, "--device", std::getenv("LOCAL_RANK") ? std::stoll(std::getenv("LOCAL_RANK")) : global_rank);
    const int64_t layers = arg_i64(argc, argv, "--layers", 2);
    const int64_t prompt_len = arg_i64(argc, argv, "--prompt-len", 2);
    const int64_t response_len = arg_i64(argc, argv, "--response-len", 2);
    const int64_t micro_batches = arg_i64(argc, argv, "--micro-batches", pp_size);
    const int64_t steps = arg_i64(argc, argv, "--steps", 1);
    const double lr = arg_f64(argc, argv, "--lr", 1.0e-8);
    const double clip_ratio = arg_f64(argc, argv, "--clip-ratio", 0.2);
    const double advantage_scale = arg_f64(argc, argv, "--advantage-scale", 1.0);
    const auto dtype = parse_dtype(arg_str(argc, argv, "--dtype", "bfloat16"));
    const std::string id_prefix = arg_str(argc, argv, "--id-prefix", "/tmp/cverl_qwen_pp_tp_ppo");

    if (pp_size <= 0 || tp_size <= 0 || micro_batches <= 0 || steps <= 0) {
      throw std::invalid_argument("pp-size, tp-size, micro-batches, and steps must be positive");
    }
    if (prompt_len <= 0 || response_len <= 0) {
      throw std::invalid_argument("prompt-len and response-len must be positive");
    }

    const int64_t seq_len = prompt_len + response_len;
    const auto device = torch::Device(torch::kCUDA, static_cast<int>(device_idx));

    cverl::distributed::ParallelDims dims;
    dims.data_parallel = 1;
    dims.pipeline_parallel = pp_size;
    dims.context_parallel = 1;
    dims.tensor_parallel = tp_size;
    dims.micro_batches = micro_batches;
    cverl::distributed::ClusterSpec spec;
    spec.world_size = world_size;
    spec.rank = global_rank;
    spec.local_world_size = world_size;
    spec.local_rank = device_idx;
    spec.gpus_per_node = world_size;
    spec.parallel = dims;
    cverl::distributed::Topology topology(spec);
    auto info = topology.local_rank_info();
    auto range = topology.pipeline_layer_range(layers, info.pipeline_rank);
    auto schedule = cverl::distributed::pipeline_1f1b_schedule(topology, info);
    auto peers = cverl::distributed::pipeline_peers(topology, info);

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
    cverl::distributed::NcclCollectives full_comm(global_rank, world_size, static_cast<int>(device_idx), full_id);
    cverl::distributed::NcclCollectives tp_comm(info.tensor_rank, tp_size, static_cast<int>(device_idx), tp_id);
    cverl::distributed::ParallelGroup tp_group{info.tensor_rank, tp_size, full_group(tp_size), &tp_comm};

    cverl::HfModelLoader loader(model_dir);
    cverl::Qwen35TextModel model(std::move(loader));
    model.to(device);
    std::vector<torch::Tensor> params;
    for (const auto& name : model.required_weight_names(layers)) {
      auto tensor = model.loader()
                        .load_tensor(name)
                        .to(torch::TensorOptions().device(device).dtype(dtype))
                        .contiguous();
      tensor.set_requires_grad(true);
      model.set_weight_override(name, tensor);
      params.push_back(tensor);
    }
    torch::optim::AdamW optimizer(
        params,
        torch::optim::AdamWOptions(lr)
            .betas(std::make_tuple(0.9, 0.95))
            .eps(1.0e-5)
            .weight_decay(0.01));

    const std::vector<int64_t> hidden_shape{1, seq_len, model.config().hidden_size};
    const auto full_ranks = full_group(world_size);
    bool trainer_ok = true;

    for (int64_t step = 0; step < steps; ++step) {
      optimizer.zero_grad();
      auto before = clone_params(params);
      std::unordered_map<int64_t, torch::Tensor> stage_inputs;
      std::unordered_map<int64_t, torch::Tensor> stage_outputs;
      double loss_sum = 0.0;
      double kl_sum = 0.0;
      double clipfrac_sum = 0.0;

      for (const auto& action : schedule) {
        if (action.op == cverl::distributed::PipelineScheduleOp::Forward) {
          torch::Tensor input;
          if (peers.is_first_stage) {
            auto ids = make_token_ids(seq_len, action.micro_batch, step, model.config(), device);
            input = model.token_embeddings(ids);
          } else {
            auto like = torch::empty(hidden_shape, torch::TensorOptions().device(device).dtype(dtype));
            input = full_comm.recv_like(like, peers.previous_rank).detach().set_requires_grad(true);
            stage_inputs[action.micro_batch] = input;
          }
          auto output = model.forward_hidden_range_tensor_parallel(
              input, range.begin, range.end, tp_group, peers.is_last_stage);
          stage_outputs[action.micro_batch] = output;
          if (!peers.is_last_stage) {
            full_comm.send(output.contiguous(), peers.next_rank);
          }
        } else {
          auto out_it = stage_outputs.find(action.micro_batch);
          if (out_it == stage_outputs.end()) {
            throw std::runtime_error("missing stage activation for backward");
          }
          if (peers.is_last_stage) {
            auto ids = make_token_ids(seq_len, action.micro_batch, step, model.config(), device);
            auto logits = model.lm_head_logits(out_it->second).to(torch::kFloat32);
            auto response_logits = logits.slice(1, prompt_len - 1, prompt_len + response_len - 1);
            auto response_ids = ids.slice(1, prompt_len, prompt_len + response_len);
            auto log_probs = cverl::torch_backend::response_log_probs(response_logits, response_ids);
            auto old_log_probs = log_probs.detach();
            auto advantages = torch::ones_like(log_probs) * advantage_scale;
            auto response_mask = torch::ones_like(log_probs);
            auto loss = cverl::torch_backend::ppo_clipped_loss(
                old_log_probs,
                log_probs,
                advantages,
                response_mask,
                clip_ratio,
                -1.0,
                -1.0,
                3.0,
                CVERL_LOSS_AGG_TOKEN_MEAN);
            auto scaled_loss = loss.pg_loss / static_cast<double>(micro_batches);
            loss_sum += scaled_loss.item<double>();
            kl_sum += loss.ppo_kl.item<double>();
            clipfrac_sum += loss.pg_clipfrac.item<double>();
            scaled_loss.backward();
          } else {
            auto grad = full_comm.recv_like(out_it->second.contiguous(), peers.next_rank);
            out_it->second.backward(grad);
          }

          if (!peers.is_first_stage) {
            auto in_it = stage_inputs.find(action.micro_batch);
            if (in_it == stage_inputs.end()) {
              throw std::runtime_error("missing stage input for backward");
            }
            full_comm.send(in_it->second.grad().contiguous(), peers.previous_rank);
            stage_inputs.erase(in_it);
          }
          stage_outputs.erase(out_it);
        }
      }

      const double local_grad_norm = grad_norm_sum(params);
      optimizer.step();
      const double local_param_delta = param_delta_sum(before, params);

      auto grad_tensor = torch::tensor({local_grad_norm}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
      auto delta_tensor = torch::tensor({local_param_delta}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
      auto loss_tensor = torch::tensor({loss_sum}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
      auto kl_tensor = torch::tensor({kl_sum}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
      auto clip_tensor = torch::tensor({clipfrac_sum}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
      auto grad_norms = full_comm.all_gather(grad_tensor.contiguous(), full_ranks, 0).cpu();
      auto param_deltas = full_comm.all_gather(delta_tensor.contiguous(), full_ranks, 0).cpu();
      auto loss_sums = full_comm.all_gather(loss_tensor.contiguous(), full_ranks, 0).cpu();
      auto kl_sums = full_comm.all_gather(kl_tensor.contiguous(), full_ranks, 0).cpu();
      auto clip_sums = full_comm.all_gather(clip_tensor.contiguous(), full_ranks, 0).cpu();
      full_comm.barrier();

      if (global_rank == 0) {
        bool all_have_grad = true;
        bool all_updated = true;
        bool any_updated = false;
        double total_loss = 0.0;
        double total_kl = 0.0;
        double total_clipfrac = 0.0;
        double total_grad_norm = 0.0;
        double total_param_delta = 0.0;
        for (int64_t i = 0; i < grad_norms.numel(); ++i) {
          const double grad_value = grad_norms[i].item<double>();
          const double delta_value = param_deltas[i].item<double>();
          all_have_grad = all_have_grad && std::isfinite(grad_value) && grad_value > 0.0;
          all_updated = all_updated && std::isfinite(delta_value) && delta_value > 0.0;
          any_updated = any_updated || (std::isfinite(delta_value) && delta_value > 0.0);
          total_grad_norm += grad_value;
          total_param_delta += delta_value;
          total_loss += loss_sums[i].item<double>();
          total_kl += kl_sums[i].item<double>();
          total_clipfrac += clip_sums[i].item<double>();
        }
        std::cout << "step=" << step
                  << " pp=" << pp_size
                  << " tp=" << tp_size
                  << " micro_batches=" << micro_batches
                  << " layers=" << layers
                  << " prompt_len=" << prompt_len
                  << " response_len=" << response_len
                  << " loss_sum=" << total_loss
                  << " ppo_kl_sum=" << total_kl
                  << " clipfrac_sum=" << total_clipfrac
                  << " grad_norm_sum=" << total_grad_norm
                  << " param_delta_sum=" << total_param_delta
                  << " all_ranks_have_grad=" << (all_have_grad ? "true" : "false")
                  << " all_ranks_updated=" << (all_updated ? "true" : "false")
                  << " any_rank_updated=" << (any_updated ? "true" : "false")
                  << "\n";
        trainer_ok = trainer_ok && all_have_grad && any_updated;
      }
    }

    full_comm.barrier();
    if (global_rank == 0) {
      std::filesystem::remove(full_id_path);
      for (int64_t pp = 0; pp < pp_size; ++pp) {
        std::filesystem::remove(id_prefix + "_tp_pp" + std::to_string(pp) + ".bin");
      }
      return trainer_ok ? 0 : 2;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "qwen3_5_pp_tp_ppo_trainer failed: " << e.what() << "\n";
    return 1;
  }
}
