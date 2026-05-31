#include "cverl/distributed/optimizer_sharding.h"

#include <algorithm>
#include <stdexcept>

namespace cverl::distributed {
namespace {

void validate_dp_rank(int64_t data_parallel, int64_t data_rank) {
  if (data_parallel <= 0 || data_rank < 0 || data_rank >= data_parallel) {
    throw std::invalid_argument("invalid data parallel rank/world size");
  }
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
  validate_dp_rank(data_parallel, data_rank);
  std::vector<torch::Tensor> flat_params;
  flat_params.reserve(parameters.size());
  int64_t original_numel = 0;
  torch::Device device(torch::kCPU);
  bool have_tensor = false;
  for (const auto& parameter : parameters) {
    if (!parameter.defined() || !parameter.is_contiguous()) {
      throw std::invalid_argument("flatten_parameter_shard requires contiguous defined parameters");
    }
    if (!have_tensor) {
      device = parameter.device();
      have_tensor = true;
    } else if (parameter.device() != device) {
      throw std::invalid_argument("flatten_parameter_shard requires parameters on one device");
    }
    flat_params.push_back(parameter.detach().to(torch::kFloat32).contiguous().view({parameter.numel()}));
    original_numel += parameter.numel();
  }

  const int64_t remainder = original_numel % data_parallel;
  const int64_t pad = remainder == 0 ? 0 : data_parallel - remainder;
  const int64_t padded_numel = original_numel + pad;
  torch::Tensor flat;
  if (flat_params.empty()) {
    flat = torch::zeros({padded_numel}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
  } else {
    flat = torch::cat(flat_params, 0).contiguous();
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

  int64_t parameter_begin = 0;
  for (size_t i = 0; i < parameters.size(); ++i) {
    const int64_t parameter_end = parameter_begin + parameters[i].numel();
    const int64_t overlap_begin = std::max(parameter_begin, out.shard_begin);
    const int64_t overlap_end = std::min(parameter_end, out.shard_end);
    if (overlap_begin < overlap_end) {
      out.ranges.push_back(FlatParameterShardRange{
          static_cast<int64_t>(i),
          overlap_begin - parameter_begin,
          overlap_begin - out.shard_begin,
          overlap_end - overlap_begin,
      });
    }
    parameter_begin = parameter_end;
  }
  return out;
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

}  // namespace cverl::distributed
