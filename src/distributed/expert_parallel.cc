#include "cverl/distributed/expert_parallel.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <torch/csrc/autograd/custom_function.h>

namespace cverl::distributed {
namespace {

torch::Tensor group_tensor_from_vector(const std::vector<int64_t>& group) {
  return torch::tensor(group, torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
}

std::vector<int64_t> group_vector_from_tensor(const torch::Tensor& tensor) {
  auto cpu = tensor.to(torch::kCPU).contiguous();
  std::vector<int64_t> group;
  group.reserve(static_cast<size_t>(cpu.numel()));
  auto accessor = cpu.accessor<int64_t, 1>();
  for (int64_t i = 0; i < cpu.numel(); ++i) {
    group.push_back(accessor[i]);
  }
  return group;
}

int64_t normalize_dim(const torch::Tensor& input, int64_t dim) {
  if (!input.defined()) {
    throw std::invalid_argument("expert parallel all_to_all input must be defined");
  }
  int64_t normalized = dim;
  if (normalized < 0) {
    normalized += input.dim();
  }
  if (normalized < 0 || normalized >= input.dim()) {
    throw std::invalid_argument("expert parallel all_to_all dim out of range");
  }
  return normalized;
}

class ExpertParallelAllToAllFunction final : public torch::autograd::Function<ExpertParallelAllToAllFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::Tensor input,
                               torch::Tensor group_tensor,
                               int64_t collectives_addr,
                               int64_t dim) {
    auto* collectives = reinterpret_cast<Collectives*>(static_cast<uintptr_t>(collectives_addr));
    if (collectives == nullptr) {
      throw std::invalid_argument("expert_parallel_all_to_all_autograd requires collectives");
    }
    const auto group = group_vector_from_tensor(group_tensor);
    const int64_t normalized_dim = normalize_dim(input, dim);
    ctx->saved_data["collectives_addr"] = collectives_addr;
    ctx->saved_data["dim"] = normalized_dim;
    ctx->save_for_backward({group_tensor});

    auto moved = normalized_dim == 0 ? input.contiguous() : input.transpose(0, normalized_dim).contiguous();
    auto output = collectives->all_to_all(moved, group, 0).contiguous();
    return normalized_dim == 0 ? output : output.transpose(0, normalized_dim).contiguous();
  }

  static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx,
                                               torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto* collectives =
        reinterpret_cast<Collectives*>(static_cast<uintptr_t>(ctx->saved_data["collectives_addr"].toInt()));
    const int64_t dim = ctx->saved_data["dim"].toInt();
    const auto group = group_vector_from_tensor(saved.at(0));

    auto grad = grad_outputs.at(0);
    auto moved = dim == 0 ? grad.contiguous() : grad.transpose(0, dim).contiguous();
    auto input_grad = collectives->all_to_all(moved, group, 0).contiguous();
    if (dim != 0) {
      input_grad = input_grad.transpose(0, dim).contiguous();
    }
    return {input_grad, torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }
};

}  // namespace

torch::Tensor expert_parallel_all_to_all_autograd(const torch::Tensor& input,
                                                  Collectives& collectives,
                                                  const std::vector<int64_t>& group,
                                                  int64_t dim) {
  if (group.empty()) {
    throw std::invalid_argument("expert parallel all_to_all group must not be empty");
  }
  return ExpertParallelAllToAllFunction::apply(
      input, group_tensor_from_vector(group), static_cast<int64_t>(reinterpret_cast<uintptr_t>(&collectives)), dim);
}

ExpertParallelDispatchResult expert_parallel_dispatch_equal_capacity(const torch::Tensor& tokens,
                                                                     const torch::Tensor& destination_ranks,
                                                                     Collectives& collectives,
                                                                     const std::vector<int64_t>& group) {
  if (!tokens.defined() || tokens.dim() != 2) {
    throw std::invalid_argument("expert_parallel_dispatch_equal_capacity requires tokens [N,H]");
  }
  if (!destination_ranks.defined() || destination_ranks.dim() != 1 || destination_ranks.size(0) != tokens.size(0)) {
    throw std::invalid_argument("expert_parallel_dispatch_equal_capacity requires destination_ranks [N]");
  }
  if (destination_ranks.scalar_type() != torch::kInt64) {
    throw std::invalid_argument("expert_parallel_dispatch_equal_capacity destination_ranks must be int64");
  }
  if (group.empty()) {
    throw std::invalid_argument("expert_parallel_dispatch_equal_capacity group must not be empty");
  }

  const int64_t world = static_cast<int64_t>(group.size());
  const auto device = tokens.device();
  std::vector<torch::Tensor> selected_tokens;
  std::vector<torch::Tensor> payload_indices;
  std::vector<int64_t> counts_host(static_cast<size_t>(world), 0);

  int64_t local_capacity = 0;
  for (int64_t dest = 0; dest < world; ++dest) {
    auto positions = torch::nonzero(destination_ranks == dest).flatten().to(torch::TensorOptions().dtype(torch::kLong).device(device));
    const int64_t count = positions.numel();
    counts_host[static_cast<size_t>(dest)] = count;
    local_capacity = std::max(local_capacity, count);
  }

  auto counts = torch::tensor(counts_host, torch::TensorOptions().dtype(torch::kInt64)).to(device).contiguous();
  auto local_capacity_tensor =
      torch::full({1}, local_capacity, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto capacity_tensor = collectives.all_reduce(local_capacity_tensor, ReduceOp::Max, group).to(torch::kCPU).contiguous();
  const int64_t capacity = capacity_tensor.item<int64_t>();

  auto recv_counts = collectives.all_to_all(counts.contiguous(), group, 0).contiguous();
  auto payload = torch::zeros({world * capacity, tokens.size(1)}, tokens.options());
  if (capacity > 0) {
    for (int64_t dest = 0; dest < world; ++dest) {
      auto positions =
          torch::nonzero(destination_ranks == dest).flatten().to(torch::TensorOptions().dtype(torch::kLong).device(device));
      const int64_t count = positions.numel();
      if (count == 0) {
        continue;
      }
      selected_tokens.push_back(tokens.index_select(0, positions));
      payload_indices.push_back(
          torch::arange(dest * capacity, dest * capacity + count, torch::TensorOptions().dtype(torch::kLong).device(device)));
    }
    if (!selected_tokens.empty()) {
      auto all_selected = torch::cat(selected_tokens, 0).contiguous();
      auto all_indices = torch::cat(payload_indices, 0).contiguous();
      payload = payload.index_add(0, all_indices, all_selected);
    }
  }

  auto received = expert_parallel_all_to_all_autograd(payload.contiguous(), collectives, group, 0).contiguous();
  return ExpertParallelDispatchResult{received, counts, recv_counts, capacity};
}

}  // namespace cverl::distributed
