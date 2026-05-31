#include "cverl/distributed/optimizer_sharding.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cverl::distributed {
namespace {

void validate_dp_rank(int64_t data_parallel, int64_t data_rank) {
  if (data_parallel <= 0 || data_rank < 0 || data_rank >= data_parallel) {
    throw std::invalid_argument("invalid data parallel rank/world size");
  }
}

FlatParameterShard flatten_tensor_list_shard(const std::vector<torch::Tensor>& tensors,
                                             int64_t data_parallel,
                                             int64_t data_rank,
                                             const char* name) {
  validate_dp_rank(data_parallel, data_rank);
  std::vector<torch::Tensor> flat_tensors;
  flat_tensors.reserve(tensors.size());
  int64_t original_numel = 0;
  torch::Device device(torch::kCPU);
  bool have_tensor = false;
  for (const auto& tensor : tensors) {
    if (!tensor.defined() || !tensor.is_contiguous()) {
      throw std::invalid_argument(std::string(name) + " requires contiguous defined tensors");
    }
    if (!have_tensor) {
      device = tensor.device();
      have_tensor = true;
    } else if (tensor.device() != device) {
      throw std::invalid_argument(std::string(name) + " requires tensors on one device");
    }
    flat_tensors.push_back(tensor.detach().to(torch::kFloat32).contiguous().view({tensor.numel()}));
    original_numel += tensor.numel();
  }

  const int64_t remainder = original_numel % data_parallel;
  const int64_t pad = remainder == 0 ? 0 : data_parallel - remainder;
  const int64_t padded_numel = original_numel + pad;
  torch::Tensor flat;
  if (flat_tensors.empty()) {
    flat = torch::zeros({padded_numel}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
  } else {
    flat = torch::cat(flat_tensors, 0).contiguous();
    if (pad > 0) {
      flat = torch::cat({flat, torch::zeros({pad}, flat.options())}, 0).contiguous();
    }
  }

  const int64_t shard_size = padded_numel / data_parallel;
  FlatParameterShard out;
  out.original_numel = original_numel;
  out.padded_numel = padded_numel;
  out.shard_begin = data_rank * shard_size;
  out.shard_end = out.shard_begin + shard_size;
  out.shard = flat.narrow(0, out.shard_begin, shard_size).contiguous();

  int64_t tensor_begin = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    const int64_t tensor_end = tensor_begin + tensors[i].numel();
    const int64_t overlap_begin = std::max(tensor_begin, out.shard_begin);
    const int64_t overlap_end = std::min(tensor_end, out.shard_end);
    if (overlap_begin < overlap_end) {
      out.ranges.push_back(FlatParameterShardRange{
          static_cast<int64_t>(i),
          overlap_begin - tensor_begin,
          overlap_begin - out.shard_begin,
          overlap_end - overlap_begin,
      });
    }
    tensor_begin = tensor_end;
  }
  return out;
}

int64_t rank_index_in_group(int64_t rank, const std::vector<int64_t>& group, const char* name) {
  if (group.empty()) {
    throw std::invalid_argument(std::string(name) + " requires a non-empty data group");
  }
  for (size_t i = 0; i < group.size(); ++i) {
    if (group[i] == rank) {
      return static_cast<int64_t>(i);
    }
  }
  throw std::invalid_argument(std::string(name) + " collectives rank is not in data group");
}

std::vector<torch::Tensor> gradient_tensor_list(const std::vector<torch::Tensor>& parameters,
                                                bool require_grad,
                                                const char* name) {
  std::vector<torch::Tensor> gradients;
  gradients.reserve(parameters.size());
  for (const auto& parameter : parameters) {
    if (!parameter.defined()) {
      throw std::invalid_argument(std::string(name) + " requires defined parameters");
    }
    if (!parameter.grad().defined()) {
      if (require_grad) {
        throw std::invalid_argument(std::string(name) + " missing parameter gradient");
      }
      gradients.push_back(torch::zeros_like(parameter, torch::TensorOptions().dtype(torch::kFloat32)));
      continue;
    }
    auto grad = parameter.grad().detach();
    if (grad.numel() != parameter.numel()) {
      throw std::invalid_argument(std::string(name) + " gradient size mismatch");
    }
    gradients.push_back(grad.to(parameter.device(), torch::kFloat32).contiguous().view(parameter.sizes()));
  }
  return gradients;
}

}  // namespace

std::vector<int64_t> greedy_parameter_owner_by_size(const std::vector<int64_t>& parameter_bytes,
                                                    int64_t data_parallel) {
  if (data_parallel <= 0) {
    throw std::invalid_argument("data_parallel must be positive");
  }
  std::vector<int64_t> bytes_by_rank(static_cast<size_t>(data_parallel), 0);
  std::vector<int64_t> owner_by_parameter;
  owner_by_parameter.reserve(parameter_bytes.size());
  for (int64_t bytes : parameter_bytes) {
    if (bytes < 0) {
      throw std::invalid_argument("parameter byte size must be non-negative");
    }
    int64_t owner = 0;
    for (int64_t rank = 1; rank < data_parallel; ++rank) {
      if (bytes_by_rank[static_cast<size_t>(rank)] < bytes_by_rank[static_cast<size_t>(owner)]) {
        owner = rank;
      }
    }
    owner_by_parameter.push_back(owner);
    bytes_by_rank[static_cast<size_t>(owner)] += bytes;
  }
  return owner_by_parameter;
}

std::vector<int64_t> owned_parameter_indices(const std::vector<int64_t>& owner_by_parameter,
                                             int64_t data_rank) {
  if (data_rank < 0) {
    throw std::invalid_argument("data_rank must be non-negative");
  }
  std::vector<int64_t> out;
  for (size_t i = 0; i < owner_by_parameter.size(); ++i) {
    if (owner_by_parameter[i] < 0) {
      throw std::invalid_argument("owner rank must be non-negative");
    }
    if (owner_by_parameter[i] == data_rank) {
      out.push_back(static_cast<int64_t>(i));
    }
  }
  return out;
}

FlatParameterShard flatten_parameter_shard(const std::vector<torch::Tensor>& parameters,
                                           int64_t data_parallel,
                                           int64_t data_rank) {
  for (const auto& parameter : parameters) {
    if (!parameter.defined()) {
      throw std::invalid_argument("flatten_parameter_shard requires contiguous defined parameters");
    }
  }
  return flatten_tensor_list_shard(parameters, data_parallel, data_rank, "flatten_parameter_shard");
}

FlatParameterShard flatten_gradient_shard(const std::vector<torch::Tensor>& parameters,
                                          int64_t data_parallel,
                                          int64_t data_rank,
                                          bool require_grad) {
  auto gradients = gradient_tensor_list(parameters, require_grad, "flatten_gradient_shard");
  return flatten_tensor_list_shard(gradients, data_parallel, data_rank, "flatten_gradient_shard");
}

FlatParameterShard reduce_scatter_flat_gradient_shard(const std::vector<torch::Tensor>& parameters,
                                                      Collectives& collectives,
                                                      const std::vector<int64_t>& data_group,
                                                      bool average,
                                                      bool require_grad) {
  const int64_t data_rank =
      rank_index_in_group(collectives.rank(), data_group, "reduce_scatter_flat_gradient_shard");
  const int64_t data_parallel = static_cast<int64_t>(data_group.size());
  auto gradients = gradient_tensor_list(parameters, require_grad, "reduce_scatter_flat_gradient_shard");
  auto full_flat = flatten_tensor_list_shard(gradients, 1, 0, "reduce_scatter_flat_gradient_shard");
  auto metadata = flatten_tensor_list_shard(gradients, data_parallel, data_rank, "reduce_scatter_flat_gradient_shard");
  auto flat = full_flat.shard;
  if (metadata.padded_numel > flat.numel()) {
    flat = torch::cat({flat, torch::zeros({metadata.padded_numel - flat.numel()}, flat.options())}, 0).contiguous();
  }
  metadata.shard = collectives
                       .reduce_scatter(
                           flat.contiguous(), average ? ReduceOp::Mean : ReduceOp::Sum, data_group, 0)
                       .contiguous();
  if (metadata.shard.numel() != metadata.shard_end - metadata.shard_begin) {
    throw std::runtime_error("reduce_scatter_flat_gradient_shard returned an unexpected shard size");
  }
  return metadata;
}

torch::Tensor all_gather_flat_parameter_shards(const FlatParameterShard& local_shard,
                                               Collectives& collectives,
                                               const std::vector<int64_t>& data_group) {
  const int64_t data_rank =
      rank_index_in_group(collectives.rank(), data_group, "all_gather_flat_parameter_shards");
  const int64_t data_parallel = static_cast<int64_t>(data_group.size());
  if (!local_shard.shard.defined() || local_shard.shard.dim() != 1 || !local_shard.shard.is_contiguous()) {
    throw std::invalid_argument("all_gather_flat_parameter_shards requires a contiguous 1D local shard");
  }
  if (local_shard.original_numel < 0 || local_shard.padded_numel < local_shard.original_numel ||
      local_shard.padded_numel % data_parallel != 0) {
    throw std::invalid_argument("all_gather_flat_parameter_shards invalid shard numel metadata");
  }
  const int64_t shard_size = local_shard.padded_numel / data_parallel;
  if (local_shard.shard.numel() != shard_size ||
      local_shard.shard_begin != data_rank * shard_size ||
      local_shard.shard_end != local_shard.shard_begin + shard_size) {
    throw std::invalid_argument("all_gather_flat_parameter_shards local shard metadata mismatch");
  }
  auto gathered = collectives.all_gather(local_shard.shard.contiguous(), data_group, 0).contiguous();
  if (gathered.dim() != 1 || gathered.numel() != local_shard.padded_numel) {
    throw std::runtime_error("all_gather_flat_parameter_shards returned an unexpected tensor shape");
  }
  return gathered;
}

void all_gather_apply_flat_parameter_shard(const FlatParameterShard& local_shard,
                                           Collectives& collectives,
                                           const std::vector<int64_t>& data_group,
                                           const std::vector<torch::Tensor>& parameters) {
  auto gathered = all_gather_flat_parameter_shards(local_shard, collectives, data_group);
  apply_full_flat_parameters(gathered, local_shard.original_numel, parameters);
}

FlatAdamWStepResult flat_sharded_adamw_step(const std::vector<torch::Tensor>& parameters,
                                            FlatParameterShard& parameter_shard,
                                            cverl::torch_backend::FlatAdamW& optimizer,
                                            Collectives& data_collectives,
                                            const std::vector<int64_t>& data_group,
                                            Collectives& norm_collectives,
                                            const std::vector<int64_t>& norm_group,
                                            double max_grad_norm,
                                            bool average_gradients,
                                            bool require_grad,
                                            bool apply_parameters) {
  if (max_grad_norm < 0.0) {
    throw std::invalid_argument("flat_sharded_adamw_step max_grad_norm must be non-negative");
  }
  auto grad_shard = reduce_scatter_flat_gradient_shard(
      parameters, data_collectives, data_group, average_gradients, require_grad);
  if (grad_shard.original_numel != parameter_shard.original_numel ||
      grad_shard.padded_numel != parameter_shard.padded_numel ||
      grad_shard.shard_begin != parameter_shard.shard_begin ||
      grad_shard.shard_end != parameter_shard.shard_end ||
      grad_shard.shard.numel() != parameter_shard.shard.numel()) {
    throw std::runtime_error("flat_sharded_adamw_step parameter/gradient shard metadata mismatch");
  }

  FlatAdamWStepResult result;
  result.gradient_shard = std::move(grad_shard);
  result.local_grad_norm_sq = result.gradient_shard.shard.pow(2).sum().item<double>();
  auto grad_sq_tensor = torch::tensor(
      {result.local_grad_norm_sq},
      torch::TensorOptions().device(result.gradient_shard.shard.device()).dtype(torch::kFloat32));
  auto global_grad_sq_tensor =
      norm_collectives.all_reduce(grad_sq_tensor.contiguous(), ReduceOp::Sum, norm_group);
  result.global_grad_norm =
      std::sqrt(static_cast<double>(global_grad_sq_tensor.cpu()[0].item<float>()));
  if (max_grad_norm > 0.0 && std::isfinite(result.global_grad_norm) &&
      result.global_grad_norm > max_grad_norm) {
    result.grad_clip_scale = max_grad_norm / (result.global_grad_norm + 1.0e-6);
    result.gradient_shard.shard.mul_(result.grad_clip_scale);
  }
  result.local_grad_norm = result.gradient_shard.shard.norm().item<double>();
  optimizer.step(result.gradient_shard.shard);
  parameter_shard.shard.copy_(optimizer.parameter_shard());
  if (apply_parameters) {
    all_gather_apply_flat_parameter_shard(parameter_shard, data_collectives, data_group, parameters);
  }
  return result;
}

void apply_flat_parameter_shard(const FlatParameterShard& shard,
                                const std::vector<torch::Tensor>& parameters) {
  if (!shard.shard.defined() || shard.shard.dim() != 1 || !shard.shard.is_contiguous()) {
    throw std::invalid_argument("apply_flat_parameter_shard requires a contiguous 1D shard tensor");
  }
  torch::NoGradGuard no_grad;
  for (const auto& range : shard.ranges) {
    if (range.parameter_index < 0 || range.parameter_index >= static_cast<int64_t>(parameters.size()) ||
        range.parameter_offset < 0 || range.shard_offset < 0 || range.numel < 0 ||
        range.shard_offset + range.numel > shard.shard.numel()) {
      throw std::invalid_argument("invalid flat parameter shard range");
    }
    auto& parameter = parameters[static_cast<size_t>(range.parameter_index)];
    if (!parameter.defined() || !parameter.is_contiguous() ||
        range.parameter_offset + range.numel > parameter.numel()) {
      throw std::invalid_argument("flat parameter shard range exceeds parameter");
    }
    auto source = shard.shard.narrow(0, range.shard_offset, range.numel)
                      .to(parameter.device(), parameter.scalar_type());
    parameter.view({parameter.numel()}).narrow(0, range.parameter_offset, range.numel).copy_(source);
  }
}

void apply_full_flat_parameters(const torch::Tensor& flat_parameters,
                                int64_t original_numel,
                                const std::vector<torch::Tensor>& parameters) {
  if (!flat_parameters.defined() || flat_parameters.dim() != 1 || !flat_parameters.is_contiguous()) {
    throw std::invalid_argument("apply_full_flat_parameters requires a contiguous 1D tensor");
  }
  if (original_numel < 0 || original_numel > flat_parameters.numel()) {
    throw std::invalid_argument("apply_full_flat_parameters original_numel out of range");
  }
  int64_t expected_numel = 0;
  for (const auto& parameter : parameters) {
    if (!parameter.defined() || !parameter.is_contiguous()) {
      throw std::invalid_argument("apply_full_flat_parameters requires contiguous defined parameters");
    }
    expected_numel += parameter.numel();
  }
  if (expected_numel != original_numel) {
    throw std::invalid_argument("apply_full_flat_parameters original_numel does not match parameters");
  }

  torch::NoGradGuard no_grad;
  int64_t offset = 0;
  for (const auto& parameter : parameters) {
    const int64_t numel = parameter.numel();
    auto source = flat_parameters.narrow(0, offset, numel)
                      .to(parameter.device(), parameter.scalar_type());
    parameter.view({numel}).copy_(source);
    offset += numel;
  }
}

}  // namespace cverl::distributed
