#include "cverl/distributed/optimizer_sharding.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
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
                                             torch::ScalarType shard_dtype,
                                             const char* name) {
  validate_dp_rank(data_parallel, data_rank);
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
    original_numel += tensor.numel();
  }

  const int64_t remainder = original_numel % data_parallel;
  const int64_t pad = remainder == 0 ? 0 : data_parallel - remainder;
  const int64_t padded_numel = original_numel + pad;
  const int64_t shard_size = padded_numel / data_parallel;
  FlatParameterShard out;
  out.original_numel = original_numel;
  out.padded_numel = padded_numel;
  out.shard_begin = data_rank * shard_size;
  out.shard_end = out.shard_begin + shard_size;

  std::vector<torch::Tensor> shard_parts;
  shard_parts.reserve(tensors.size() + 1);
  const auto shard_options = torch::TensorOptions().device(device).dtype(shard_dtype);
  int64_t tensor_begin = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    const int64_t tensor_end = tensor_begin + tensors[i].numel();
    const int64_t overlap_begin = std::max(tensor_begin, out.shard_begin);
    const int64_t overlap_end = std::min(tensor_end, out.shard_end);
    if (overlap_begin < overlap_end) {
      auto part = tensors[i].detach()
                      .view({tensors[i].numel()})
                      .narrow(0, overlap_begin - tensor_begin, overlap_end - overlap_begin)
                      .to(shard_options.device(), shard_options.dtype())
                      .contiguous();
      shard_parts.push_back(part);
      out.ranges.push_back(FlatParameterShardRange{
          static_cast<int64_t>(i),
          overlap_begin - tensor_begin,
          overlap_begin - out.shard_begin,
          overlap_end - overlap_begin,
      });
    }
    tensor_begin = tensor_end;
  }
  const int64_t valid_begin = std::min(out.shard_begin, original_numel);
  const int64_t valid_end = std::min(out.shard_end, original_numel);
  const int64_t valid_numel = std::max<int64_t>(0, valid_end - valid_begin);
  const int64_t pad_numel = shard_size - valid_numel;
  if (pad_numel > 0) {
    shard_parts.push_back(torch::zeros({pad_numel}, shard_options));
  }
  if (shard_parts.empty()) {
    out.shard = torch::zeros({shard_size}, shard_options);
  } else if (shard_parts.size() == 1) {
    out.shard = shard_parts[0].contiguous();
  } else {
    out.shard = torch::cat(shard_parts, 0).contiguous();
  }
  return out;
}

FlatParameterShard flatten_tensor_list_shard(const std::vector<torch::Tensor>& tensors,
                                             int64_t data_parallel,
                                             int64_t data_rank,
                                             const char* name) {
  return flatten_tensor_list_shard(tensors, data_parallel, data_rank, torch::kFloat32, name);
}

FlatParameterShard make_flat_shard_metadata(const std::vector<int64_t>& tensor_numels,
                                            int64_t data_parallel,
                                            int64_t data_rank,
                                            torch::Tensor shard,
                                            const char* name) {
  validate_dp_rank(data_parallel, data_rank);
  int64_t original_numel = 0;
  for (int64_t numel : tensor_numels) {
    if (numel < 0) {
      throw std::invalid_argument(std::string(name) + " requires non-negative tensor numel");
    }
    original_numel += numel;
  }
  const int64_t remainder = original_numel % data_parallel;
  const int64_t pad = remainder == 0 ? 0 : data_parallel - remainder;
  const int64_t padded_numel = original_numel + pad;
  const int64_t shard_size = padded_numel / data_parallel;
  if (!shard.defined() || shard.dim() != 1 || !shard.is_contiguous() || shard.numel() != shard_size) {
    throw std::invalid_argument(std::string(name) + " requires a contiguous 1D shard with matching size");
  }

  FlatParameterShard out;
  out.shard = std::move(shard);
  out.original_numel = original_numel;
  out.padded_numel = padded_numel;
  out.shard_begin = data_rank * shard_size;
  out.shard_end = out.shard_begin + shard_size;

  int64_t tensor_begin = 0;
  for (size_t i = 0; i < tensor_numels.size(); ++i) {
    const int64_t tensor_end = tensor_begin + tensor_numels[i];
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

struct FlatShardGroupInfo {
  int64_t data_rank = 0;
  int64_t data_parallel = 1;
  int64_t shard_size = 0;
};

FlatShardGroupInfo validate_flat_shard_for_group(const FlatParameterShard& local_shard,
                                                 int64_t collectives_rank,
                                                 const std::vector<int64_t>& data_group,
                                                 const char* name) {
  FlatShardGroupInfo info;
  info.data_rank = rank_index_in_group(collectives_rank, data_group, name);
  info.data_parallel = static_cast<int64_t>(data_group.size());
  if (!local_shard.shard.defined() || local_shard.shard.dim() != 1 || !local_shard.shard.is_contiguous()) {
    throw std::invalid_argument(std::string(name) + " requires a contiguous 1D local shard");
  }
  if (local_shard.original_numel < 0 || local_shard.padded_numel < local_shard.original_numel ||
      local_shard.padded_numel % info.data_parallel != 0) {
    throw std::invalid_argument(std::string(name) + " invalid shard numel metadata");
  }
  info.shard_size = local_shard.padded_numel / info.data_parallel;
  if (local_shard.shard.numel() != info.shard_size ||
      local_shard.shard_begin != info.data_rank * info.shard_size ||
      local_shard.shard_end != local_shard.shard_begin + info.shard_size) {
    throw std::invalid_argument(std::string(name) + " local shard metadata mismatch");
  }
  return info;
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

std::vector<int64_t> tensor_numels(const std::vector<torch::Tensor>& tensors, const char* name) {
  std::vector<int64_t> out;
  out.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    if (!tensor.defined()) {
      throw std::invalid_argument(std::string(name) + " requires defined tensors");
    }
    out.push_back(tensor.numel());
  }
  return out;
}

torch::Tensor flat_tensor_range(const std::vector<torch::Tensor>& tensors,
                                int64_t original_numel,
                                int64_t begin,
                                int64_t numel,
                                const torch::TensorOptions& options) {
  if (begin < 0 || numel < 0 || begin + numel > original_numel) {
    throw std::invalid_argument("flat_tensor_range out of bounds");
  }
  std::vector<torch::Tensor> parts;
  parts.reserve(tensors.size() + 1);
  int64_t tensor_begin = 0;
  int64_t remaining_begin = begin;
  int64_t remaining_end = begin + numel;
  for (const auto& tensor : tensors) {
    const int64_t tensor_end = tensor_begin + tensor.numel();
    const int64_t overlap_begin = std::max(tensor_begin, remaining_begin);
    const int64_t overlap_end = std::min(tensor_end, remaining_end);
    if (overlap_begin < overlap_end) {
      parts.push_back(tensor.view({tensor.numel()})
                          .narrow(0, overlap_begin - tensor_begin, overlap_end - overlap_begin)
                          .to(options.device(), options.dtype())
                          .contiguous());
    }
    tensor_begin = tensor_end;
    if (tensor_begin >= remaining_end) {
      break;
    }
  }
  if (parts.empty()) {
    return torch::empty({0}, options);
  }
  if (parts.size() == 1) {
    return parts[0].contiguous();
  }
  return torch::cat(parts, 0).contiguous();
}

torch::Tensor flat_tensor_range_padded(const std::vector<torch::Tensor>& tensors,
                                       int64_t original_numel,
                                       int64_t padded_numel,
                                       int64_t begin,
                                       int64_t numel,
                                       const torch::TensorOptions& options) {
  if (begin < 0 || numel < 0 || begin + numel > padded_numel) {
    throw std::invalid_argument("flat_tensor_range_padded out of bounds");
  }
  const int64_t valid_len = begin < original_numel ? std::min(numel, original_numel - begin) : 0;
  torch::Tensor valid = valid_len > 0
                            ? flat_tensor_range(tensors, original_numel, begin, valid_len, options)
                            : torch::empty({0}, options);
  const int64_t suffix_pad = numel - valid_len;
  std::vector<torch::Tensor> parts;
  parts.reserve(2);
  if (valid.numel() > 0) {
    parts.push_back(valid);
  }
  if (suffix_pad > 0) {
    parts.push_back(torch::zeros({suffix_pad}, options));
  }
  if (parts.empty()) {
    return torch::zeros({numel}, options);
  }
  return parts.size() == 1 ? parts[0].contiguous() : torch::cat(parts, 0).contiguous();
}

torch::Tensor flat_gradient_range_padded(const std::vector<torch::Tensor>& parameters,
                                         int64_t original_numel,
                                         int64_t padded_numel,
                                         int64_t begin,
                                         int64_t numel,
                                         const torch::TensorOptions& options,
                                         bool require_grad,
                                         const char* name) {
  if (begin < 0 || numel < 0 || begin + numel > padded_numel) {
    throw std::invalid_argument("flat_gradient_range_padded out of bounds");
  }
  auto out = torch::zeros({numel}, options);
  if (begin >= original_numel) {
    return out;
  }
  const int64_t range_end = std::min(begin + numel, original_numel);
  int64_t parameter_begin = 0;
  for (const auto& parameter : parameters) {
    if (!parameter.defined()) {
      throw std::invalid_argument(std::string(name) + " requires defined parameters");
    }
    const int64_t parameter_end = parameter_begin + parameter.numel();
    const int64_t overlap_begin = std::max(parameter_begin, begin);
    const int64_t overlap_end = std::min(parameter_end, range_end);
    if (overlap_begin < overlap_end) {
      if (!parameter.grad().defined()) {
        if (require_grad) {
          throw std::invalid_argument(std::string(name) + " missing parameter gradient");
        }
      } else {
        auto grad = parameter.grad().detach();
        if (grad.numel() != parameter.numel()) {
          throw std::invalid_argument(std::string(name) + " gradient size mismatch");
        }
        auto grad_flat = grad.is_contiguous() ? grad.view({parameter.numel()})
                                              : grad.contiguous().view({parameter.numel()});
        const int64_t source_offset = overlap_begin - parameter_begin;
        const int64_t target_offset = overlap_begin - begin;
        const int64_t overlap_numel = overlap_end - overlap_begin;
        auto source = grad_flat.narrow(0, source_offset, overlap_numel)
                          .to(options.device(), options.dtype())
                          .contiguous();
        out.narrow(0, target_offset, overlap_numel).copy_(source);
      }
    }
    parameter_begin = parameter_end;
    if (parameter_begin >= range_end) {
      break;
    }
  }
  return out;
}

void clear_parameter_gradients(const std::vector<torch::Tensor>& parameters) {
  torch::NoGradGuard no_grad;
  for (const auto& parameter : parameters) {
    if (parameter.defined() && parameter.grad().defined()) {
      parameter.mutable_grad() = torch::Tensor();
    }
  }
}

void apply_flat_parameter_range(const torch::Tensor& flat_slice,
                                int64_t global_begin,
                                int64_t original_numel,
                                const std::vector<torch::Tensor>& parameters) {
  if (!flat_slice.defined() || flat_slice.dim() != 1 || !flat_slice.is_contiguous()) {
    throw std::invalid_argument("apply_flat_parameter_range requires a contiguous 1D tensor");
  }
  if (global_begin < 0 || original_numel < 0) {
    throw std::invalid_argument("apply_flat_parameter_range invalid offsets");
  }
  const int64_t global_end = global_begin + flat_slice.numel();
  if (global_begin >= original_numel) {
    return;
  }
  torch::NoGradGuard no_grad;
  int64_t parameter_begin = 0;
  for (const auto& parameter : parameters) {
    if (!parameter.defined() || !parameter.is_contiguous()) {
      throw std::invalid_argument("apply_flat_parameter_range requires contiguous defined parameters");
    }
    const int64_t parameter_end = parameter_begin + parameter.numel();
    const int64_t overlap_begin = std::max(parameter_begin, global_begin);
    const int64_t overlap_end = std::min({parameter_end, global_end, original_numel});
    if (overlap_begin < overlap_end) {
      const int64_t source_offset = overlap_begin - global_begin;
      const int64_t parameter_offset = overlap_begin - parameter_begin;
      const int64_t numel = overlap_end - overlap_begin;
      auto source = flat_slice.narrow(0, source_offset, numel)
                        .to(parameter.device(), parameter.scalar_type());
      parameter.view({parameter.numel()}).narrow(0, parameter_offset, numel).copy_(source);
    }
    parameter_begin = parameter_end;
    if (parameter_begin >= global_end) {
      break;
    }
  }
}

}  // namespace

std::vector<int64_t> greedy_parameter_owner_by_size(const std::vector<int64_t>& parameter_bytes,
                                                    int64_t data_parallel) {
  if (data_parallel <= 0) {
    throw std::invalid_argument("data_parallel must be positive");
  }
  std::vector<int64_t> bytes_by_rank(static_cast<size_t>(data_parallel), 0);
  std::vector<int64_t> owner_by_parameter(parameter_bytes.size(), 0);
  std::vector<size_t> order(parameter_bytes.size());
  std::iota(order.begin(), order.end(), 0);
  for (int64_t bytes : parameter_bytes) {
    if (bytes < 0) {
      throw std::invalid_argument("parameter byte size must be non-negative");
    }
  }
  std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
    return parameter_bytes[lhs] > parameter_bytes[rhs];
  });
  for (size_t parameter_index : order) {
    int64_t owner = 0;
    for (int64_t rank = 1; rank < data_parallel; ++rank) {
      if (bytes_by_rank[static_cast<size_t>(rank)] < bytes_by_rank[static_cast<size_t>(owner)]) {
        owner = rank;
      }
    }
    owner_by_parameter[parameter_index] = owner;
    bytes_by_rank[static_cast<size_t>(owner)] += parameter_bytes[parameter_index];
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
  return flatten_parameter_shard(parameters, data_parallel, data_rank, torch::kFloat32);
}

FlatParameterShard flatten_parameter_shard(const std::vector<torch::Tensor>& parameters,
                                           int64_t data_parallel,
                                           int64_t data_rank,
                                           torch::ScalarType shard_dtype) {
  for (const auto& parameter : parameters) {
    if (!parameter.defined()) {
      throw std::invalid_argument("flatten_parameter_shard requires contiguous defined parameters");
    }
  }
  return flatten_tensor_list_shard(
      parameters, data_parallel, data_rank, shard_dtype, "flatten_parameter_shard");
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
                                                      bool require_grad,
                                                      std::optional<torch::ScalarType> communication_dtype) {
  const int64_t data_rank =
      rank_index_in_group(collectives.rank(), data_group, "reduce_scatter_flat_gradient_shard");
  const int64_t data_parallel = static_cast<int64_t>(data_group.size());
  if (data_parallel == 1) {
    if (!communication_dtype.has_value()) {
      return flatten_gradient_shard(parameters, 1, data_rank, require_grad);
    }
    auto gradients = gradient_tensor_list(parameters, require_grad, "reduce_scatter_flat_gradient_shard");
    return flatten_tensor_list_shard(
        gradients, 1, data_rank, *communication_dtype, "reduce_scatter_flat_gradient_shard");
  }
  auto gradients = gradient_tensor_list(parameters, require_grad, "reduce_scatter_flat_gradient_shard");
  if (communication_dtype.has_value()) {
    for (auto& grad : gradients) {
      grad = grad.to(*communication_dtype).contiguous();
    }
  }
  auto numels = tensor_numels(gradients, "reduce_scatter_flat_gradient_shard");
  auto full_flat = communication_dtype.has_value()
                       ? flatten_tensor_list_shard(
                             gradients, 1, 0, *communication_dtype, "reduce_scatter_flat_gradient_shard")
                       : flatten_tensor_list_shard(gradients, 1, 0, "reduce_scatter_flat_gradient_shard");
  auto flat = full_flat.shard;
  const int64_t remainder = full_flat.original_numel % data_parallel;
  const int64_t padded_numel =
      full_flat.original_numel + (remainder == 0 ? 0 : data_parallel - remainder);
  if (padded_numel > flat.numel()) {
    flat = torch::cat({flat, torch::zeros({padded_numel - flat.numel()}, flat.options())}, 0).contiguous();
  }
  auto reduced_shard =
      collectives.reduce_scatter(flat.contiguous(), average ? ReduceOp::Mean : ReduceOp::Sum, data_group, 0)
          .contiguous();
  auto metadata = make_flat_shard_metadata(
      numels, data_parallel, data_rank, reduced_shard, "reduce_scatter_flat_gradient_shard");
  if (metadata.shard.numel() != metadata.shard_end - metadata.shard_begin) {
    throw std::runtime_error("reduce_scatter_flat_gradient_shard returned an unexpected shard size");
  }
  return metadata;
}

FlatParameterShard reduce_scatter_flat_gradient_shard_bucketed(
    const std::vector<torch::Tensor>& parameters,
    Collectives& collectives,
    const std::vector<int64_t>& data_group,
    bool average,
    int64_t bucket_numel,
    bool require_grad,
    std::optional<torch::ScalarType> communication_dtype) {
  if (bucket_numel <= 0) {
    throw std::invalid_argument("reduce_scatter_flat_gradient_shard_bucketed bucket_numel must be positive");
  }
  const int64_t data_rank =
      rank_index_in_group(collectives.rank(), data_group, "reduce_scatter_flat_gradient_shard_bucketed");
  const int64_t data_parallel = static_cast<int64_t>(data_group.size());
  std::vector<int64_t> tensor_numels;
  tensor_numels.reserve(parameters.size());
  int64_t original_numel = 0;
  torch::Device device(torch::kCPU);
  bool have_tensor = false;
  for (const auto& parameter : parameters) {
    if (!parameter.defined()) {
      throw std::invalid_argument("reduce_scatter_flat_gradient_shard_bucketed requires defined parameters");
    }
    tensor_numels.push_back(parameter.numel());
    original_numel += parameter.numel();
    if (!have_tensor) {
      device = parameter.device();
      have_tensor = true;
    } else if (parameter.device() != device) {
      throw std::invalid_argument("reduce_scatter_flat_gradient_shard_bucketed requires parameters on one device");
    }
  }
  auto options = torch::TensorOptions().device(device).dtype(communication_dtype.value_or(torch::kFloat32));
  const int64_t remainder = original_numel % data_parallel;
  const int64_t padded_numel = original_numel + (remainder == 0 ? 0 : data_parallel - remainder);
  const int64_t shard_size = padded_numel / data_parallel;
  if (data_parallel == 1) {
    auto shard = flat_gradient_range_padded(parameters,
                                            original_numel,
                                            padded_numel,
                                            0,
                                            shard_size,
                                            options,
                                            require_grad,
                                            "reduce_scatter_flat_gradient_shard_bucketed")
                     .contiguous();
    return make_flat_shard_metadata(
        tensor_numels, data_parallel, data_rank, shard, "reduce_scatter_flat_gradient_shard_bucketed");
  }
  const int64_t shard_bucket_numel = std::max<int64_t>(1, bucket_numel / data_parallel);
  auto shard = torch::empty({shard_size}, options);

  for (int64_t shard_offset = 0; shard_offset < shard_size; shard_offset += shard_bucket_numel) {
    const int64_t chunk = std::min(shard_bucket_numel, shard_size - shard_offset);
    auto input = torch::empty({chunk * data_parallel}, options);
    for (int64_t rank = 0; rank < data_parallel; ++rank) {
      const int64_t global_begin = rank * shard_size + shard_offset;
      auto segment = flat_gradient_range_padded(parameters,
                                                original_numel,
                                                padded_numel,
                                                global_begin,
                                                chunk,
                                                options,
                                                require_grad,
                                                "reduce_scatter_flat_gradient_shard_bucketed");
      input.narrow(0, rank * chunk, chunk).copy_(segment);
    }
    auto reduced =
        collectives.reduce_scatter(input, average ? ReduceOp::Mean : ReduceOp::Sum, data_group, 0).contiguous();
    if (reduced.dim() != 1 || reduced.numel() != chunk) {
      throw std::runtime_error("reduce_scatter_flat_gradient_shard_bucketed returned an unexpected chunk shape");
    }
    shard.narrow(0, shard_offset, chunk).copy_(reduced);
  }
  return make_flat_shard_metadata(
      tensor_numels, data_parallel, data_rank, shard, "reduce_scatter_flat_gradient_shard_bucketed");
}

torch::Tensor all_gather_flat_parameter_shards(const FlatParameterShard& local_shard,
                                               Collectives& collectives,
                                               const std::vector<int64_t>& data_group) {
  const auto info = validate_flat_shard_for_group(
      local_shard, collectives.rank(), data_group, "all_gather_flat_parameter_shards");
  if (info.data_parallel == 1) {
    return local_shard.shard.contiguous();
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
  const auto info = validate_flat_shard_for_group(
      local_shard, collectives.rank(), data_group, "all_gather_apply_flat_parameter_shard");
  if (info.data_parallel == 1) {
    apply_flat_parameter_shard(local_shard, parameters);
    return;
  }
  auto gathered = all_gather_flat_parameter_shards(local_shard, collectives, data_group);
  apply_full_flat_parameters(gathered, local_shard.original_numel, parameters);
}

void all_gather_apply_flat_parameter_shard_bucketed(const FlatParameterShard& local_shard,
                                                    Collectives& collectives,
                                                    const std::vector<int64_t>& data_group,
                                                    const std::vector<torch::Tensor>& parameters,
                                                    int64_t bucket_numel) {
  if (bucket_numel <= 0) {
    throw std::invalid_argument("all_gather_apply_flat_parameter_shard_bucketed bucket_numel must be positive");
  }
  const auto info = validate_flat_shard_for_group(
      local_shard, collectives.rank(), data_group, "all_gather_apply_flat_parameter_shard_bucketed");
  const int64_t data_parallel = info.data_parallel;
  const int64_t shard_size = info.shard_size;
  if (info.data_parallel == 1) {
    apply_flat_parameter_shard(local_shard, parameters);
    return;
  }
  const int64_t shard_bucket_numel = std::max<int64_t>(1, bucket_numel / data_parallel);
  for (int64_t shard_offset = 0; shard_offset < shard_size; shard_offset += shard_bucket_numel) {
    const int64_t chunk = std::min(shard_bucket_numel, shard_size - shard_offset);
    auto local_chunk = local_shard.shard.narrow(0, shard_offset, chunk).contiguous();
    auto gathered = collectives.all_gather(local_chunk, data_group, 0).contiguous();
    if (gathered.dim() != 1 || gathered.numel() != chunk * data_parallel) {
      throw std::runtime_error("all_gather_apply_flat_parameter_shard_bucketed returned unexpected tensor shape");
    }
    for (int64_t rank = 0; rank < data_parallel; ++rank) {
      const int64_t global_begin = rank * shard_size + shard_offset;
      auto rank_slice = gathered.narrow(0, rank * chunk, chunk).contiguous();
      apply_flat_parameter_range(rank_slice, global_begin, local_shard.original_numel, parameters);
    }
  }
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
                                            bool apply_parameters,
                                            int64_t reduce_scatter_bucket_numel,
                                            int64_t all_gather_bucket_numel,
                                            std::optional<torch::ScalarType> gradient_communication_dtype) {
  if (max_grad_norm < 0.0) {
    throw std::invalid_argument("flat_sharded_adamw_step max_grad_norm must be non-negative");
  }
  auto grad_shard = reduce_scatter_bucket_numel > 0
                        ? reduce_scatter_flat_gradient_shard_bucketed(parameters,
                                                                      data_collectives,
                                                                      data_group,
                                                                      average_gradients,
                                                                      reduce_scatter_bucket_numel,
                                                                      require_grad,
                                                                      gradient_communication_dtype)
                        : reduce_scatter_flat_gradient_shard(
                              parameters,
                              data_collectives,
                              data_group,
                              average_gradients,
                              require_grad,
                              gradient_communication_dtype);
  if (grad_shard.original_numel != parameter_shard.original_numel ||
      grad_shard.padded_numel != parameter_shard.padded_numel ||
      grad_shard.shard_begin != parameter_shard.shard_begin ||
      grad_shard.shard_end != parameter_shard.shard_end ||
      grad_shard.shard.numel() != parameter_shard.shard.numel()) {
    throw std::runtime_error("flat_sharded_adamw_step parameter/gradient shard metadata mismatch");
  }

  FlatAdamWStepResult result;
  result.gradient_shard = std::move(grad_shard);
  clear_parameter_gradients(parameters);
  auto grad_sq_tensor = result.gradient_shard.shard.pow(2).sum().to(torch::kFloat32).reshape({1});
  torch::Tensor global_grad_sq_tensor;
  if (norm_group.size() == 1) {
    (void)rank_index_in_group(norm_collectives.rank(), norm_group, "flat_sharded_adamw_step norm_group");
    global_grad_sq_tensor = grad_sq_tensor.contiguous();
  } else {
    global_grad_sq_tensor =
        norm_collectives.all_reduce(grad_sq_tensor.contiguous(), ReduceOp::Sum, norm_group).contiguous();
  }
  auto clip_scale_tensor = torch::ones_like(global_grad_sq_tensor);
  if (max_grad_norm > 0.0) {
    auto global_norm_tensor = global_grad_sq_tensor.sqrt();
    auto should_clip = torch::logical_and(torch::isfinite(global_norm_tensor), global_norm_tensor > max_grad_norm);
    clip_scale_tensor = torch::where(
        should_clip,
        torch::full_like(global_norm_tensor, max_grad_norm) / (global_norm_tensor + 1.0e-6),
        clip_scale_tensor);
    result.gradient_shard.shard.mul_(clip_scale_tensor);
  }
  optimizer.step(result.gradient_shard.shard);
  parameter_shard.shard.copy_(optimizer.parameter_shard());

  auto norm_metrics = torch::cat(
                          {grad_sq_tensor.contiguous(),
                           global_grad_sq_tensor.contiguous(),
                           clip_scale_tensor.contiguous()},
                          0)
                          .to(torch::kCPU)
                          .contiguous();
  result.local_grad_norm_sq = static_cast<double>(norm_metrics[0].item<float>());
  result.global_grad_norm = std::sqrt(static_cast<double>(norm_metrics[1].item<float>()));
  result.grad_clip_scale = static_cast<double>(norm_metrics[2].item<float>());
  result.local_grad_norm = std::sqrt(result.local_grad_norm_sq) * result.grad_clip_scale;
  if (apply_parameters) {
    if (all_gather_bucket_numel > 0) {
      all_gather_apply_flat_parameter_shard_bucketed(
          parameter_shard, data_collectives, data_group, parameters, all_gather_bucket_numel);
    } else {
      all_gather_apply_flat_parameter_shard(parameter_shard, data_collectives, data_group, parameters);
    }
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
