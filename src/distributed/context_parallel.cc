#include "cverl/distributed/context_parallel.h"

#include "cverl/distributed/cp_attention_cuda.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace cverl::distributed {
namespace {

void validate_context_rank(int64_t context_rank, int64_t context_parallel) {
  if (context_parallel <= 0 || context_rank < 0 || context_rank >= context_parallel) {
    throw std::invalid_argument("invalid context parallel rank/size");
  }
}

int64_t normalize_sequence_dim(const torch::Tensor& input, int64_t sequence_dim) {
  if (!input.defined()) {
    throw std::invalid_argument("context parallel input must be defined");
  }
  int64_t dim = sequence_dim;
  if (dim < 0) {
    dim += input.dim();
  }
  if (dim < 0 || dim >= input.dim()) {
    throw std::invalid_argument("sequence_dim out of range");
  }
  return dim;
}

void require_attention_tensor(const torch::Tensor& tensor, const char* name) {
  if (!tensor.defined() || tensor.dim() != 4) {
    throw std::invalid_argument(std::string(name) + " must be a defined 4D tensor [B,H,S,D]");
  }
}

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

torch::Tensor vector_to_cpu_i64_tensor(const std::vector<int64_t>& values) {
  return torch::tensor(values, torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
}

std::vector<int64_t> vector_from_cpu_i64_tensor(const torch::Tensor& tensor) {
  auto cpu = tensor.to(torch::kCPU).contiguous();
  std::vector<int64_t> values;
  values.reserve(static_cast<size_t>(cpu.numel()));
  auto accessor = cpu.accessor<int64_t, 1>();
  for (int64_t i = 0; i < cpu.numel(); ++i) {
    values.push_back(accessor[i]);
  }
  return values;
}

torch::Tensor streaming_attention_from_ring_blocks(const torch::Tensor& query_local,
                                                   const torch::Tensor& key_ring,
                                                   const torch::Tensor& value_ring,
                                                   const std::vector<int64_t>& key_begin_positions,
                                                   int64_t query_begin,
                                                   int64_t original_sequence_length,
                                                   int64_t shard_size,
                                                   double scale) {
  if (shard_size <= 0) {
    throw std::invalid_argument("CP ring attention shard_size must be positive");
  }
  if (key_ring.size(2) != value_ring.size(2) ||
      key_ring.size(2) < shard_size * static_cast<int64_t>(key_begin_positions.size())) {
    throw std::invalid_argument("CP ring attention KV ring shape does not match block metadata");
  }
  std::vector<torch::Tensor> key_blocks;
  std::vector<torch::Tensor> value_blocks;
  key_blocks.reserve(key_begin_positions.size());
  value_blocks.reserve(key_begin_positions.size());
  for (size_t i = 0; i < key_begin_positions.size(); ++i) {
    key_blocks.push_back(key_ring.narrow(2, static_cast<int64_t>(i) * shard_size, shard_size).contiguous());
    value_blocks.push_back(value_ring.narrow(2, static_cast<int64_t>(i) * shard_size, shard_size).contiguous());
  }
  return context_parallel_causal_attention_streaming_blocks(
      query_local, key_blocks, value_blocks, key_begin_positions, query_begin, original_sequence_length, scale);
}

class ContextParallelGatherFunction final
    : public torch::autograd::Function<ContextParallelGatherFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::Tensor local,
                               torch::Tensor group_tensor,
                               int64_t collectives_addr,
                               int64_t sequence_dim) {
    auto* collectives = reinterpret_cast<Collectives*>(static_cast<uintptr_t>(collectives_addr));
    if (collectives == nullptr) {
      throw std::invalid_argument("context_parallel_gather_autograd requires collectives");
    }
    const auto group = group_vector_from_tensor(group_tensor);
    const int64_t dim = normalize_sequence_dim(local, sequence_dim);
    ctx->saved_data["collectives_addr"] = collectives_addr;
    ctx->saved_data["sequence_dim"] = dim;
    ctx->save_for_backward({group_tensor});
    return context_parallel_gather(local, *collectives, group, dim);
  }

  static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx,
                                               torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto* collectives =
        reinterpret_cast<Collectives*>(static_cast<uintptr_t>(ctx->saved_data["collectives_addr"].toInt()));
    const int64_t dim = ctx->saved_data["sequence_dim"].toInt();
    const auto group = group_vector_from_tensor(saved.at(0));
    auto grad = grad_outputs.at(0).contiguous();
    torch::Tensor moved;
    if (dim == 0) {
      moved = grad;
    } else {
      moved = grad.transpose(0, dim).contiguous();
    }
    auto local = collectives->reduce_scatter(moved, ReduceOp::Sum, group, 0).contiguous();
    if (dim != 0) {
      local = local.transpose(0, dim).contiguous();
    }
    return {local, torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }
};

class ContextParallelRingExchangeFunction final
    : public torch::autograd::Function<ContextParallelRingExchangeFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::Tensor local,
                               torch::Tensor group_tensor,
                               int64_t collectives_addr,
                               int64_t context_rank,
                               int64_t sequence_dim) {
    auto* collectives = reinterpret_cast<Collectives*>(static_cast<uintptr_t>(collectives_addr));
    if (collectives == nullptr) {
      throw std::invalid_argument("context_parallel_ring_exchange_autograd requires collectives");
    }
    const auto group = group_vector_from_tensor(group_tensor);
    validate_context_rank(context_rank, static_cast<int64_t>(group.size()));
    const int64_t dim = normalize_sequence_dim(local, sequence_dim);
    auto current = dim == 0 ? local.contiguous() : local.transpose(0, dim).contiguous();
    const auto schedule = context_parallel_ring_schedule(group, group.at(static_cast<size_t>(context_rank)));
    std::vector<torch::Tensor> blocks;
    blocks.reserve(schedule.size());
    for (size_t i = 0; i < schedule.size(); ++i) {
      blocks.push_back(current);
      if (i + 1 < schedule.size()) {
        current = collectives->send_recv(current.contiguous(), schedule[i].send_rank, current, schedule[i].recv_rank)
                      .contiguous();
      }
    }
    ctx->saved_data["collectives_addr"] = collectives_addr;
    ctx->saved_data["context_rank"] = context_rank;
    ctx->saved_data["sequence_dim"] = dim;
    ctx->saved_data["local_shard"] = current.size(0);
    ctx->save_for_backward({group_tensor});
    auto gathered = torch::cat(blocks, 0).contiguous();
    return dim == 0 ? gathered : gathered.transpose(0, dim).contiguous();
  }

  static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx,
                                               torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto* collectives =
        reinterpret_cast<Collectives*>(static_cast<uintptr_t>(ctx->saved_data["collectives_addr"].toInt()));
    const int64_t context_rank = ctx->saved_data["context_rank"].toInt();
    const int64_t dim = ctx->saved_data["sequence_dim"].toInt();
    const int64_t shard = ctx->saved_data["local_shard"].toInt();
    const auto group = group_vector_from_tensor(saved.at(0));
    auto grad = grad_outputs.at(0).contiguous();
    auto moved = dim == 0 ? grad : grad.transpose(0, dim).contiguous();
    const auto schedule = context_parallel_ring_schedule(group, group.at(static_cast<size_t>(context_rank)));
    if (moved.size(0) != shard * static_cast<int64_t>(schedule.size())) {
      throw std::invalid_argument("ring exchange gradient shape does not match saved schedule");
    }
    std::vector<torch::Tensor> rank_order(static_cast<size_t>(group.size()));
    for (size_t i = 0; i < schedule.size(); ++i) {
      const int64_t rank_index = context_parallel_group_index(group, schedule[i].kv_rank);
      rank_order[static_cast<size_t>(rank_index)] = moved.narrow(0, static_cast<int64_t>(i) * shard, shard);
    }
    torch::Tensor local;
    auto rank_order_grad = torch::cat(rank_order, 0).contiguous();
    try {
      local = collectives->reduce_scatter(rank_order_grad, ReduceOp::Sum, group, 0).contiguous();
    } catch (const std::invalid_argument&) {
      const int64_t send_rank = schedule.empty() ? -1 : schedule.front().send_rank;
      const int64_t recv_rank = schedule.empty() ? -1 : schedule.front().recv_rank;
      for (int64_t owner = 0; owner < static_cast<int64_t>(group.size()); ++owner) {
        auto current = rank_order[static_cast<size_t>(owner)].contiguous();
        auto total = current.clone();
        for (int64_t hop = 0; hop + 1 < static_cast<int64_t>(group.size()); ++hop) {
          current = collectives->send_recv(current.contiguous(), send_rank, current, recv_rank).contiguous();
          total = total + current;
        }
        if (owner == context_rank) {
          local = total.contiguous();
        }
      }
    }
    if (!local.defined()) {
      throw std::runtime_error("ring exchange backward did not produce local owner gradient");
    }
    if (dim != 0) {
      local = local.transpose(0, dim).contiguous();
    }
    return {local, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }
};

class ContextParallelRingAttentionRecomputeFunction final
    : public torch::autograd::Function<ContextParallelRingAttentionRecomputeFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::Tensor query_local,
                               torch::Tensor key_ring,
                               torch::Tensor value_ring,
                               torch::Tensor key_begin_positions,
                               int64_t query_begin,
                               int64_t original_sequence_length,
                               int64_t shard_size,
                               double scale) {
    require_attention_tensor(query_local, "query_local");
    require_attention_tensor(key_ring, "key_ring");
    require_attention_tensor(value_ring, "value_ring");
    ctx->saved_data["query_begin"] = query_begin;
    ctx->saved_data["original_sequence_length"] = original_sequence_length;
    ctx->saved_data["shard_size"] = shard_size;
    ctx->saved_data["scale"] = scale;
    const auto positions = vector_from_cpu_i64_tensor(key_begin_positions);
    if (cp_attention_cuda_available() && query_local.is_cuda() && key_ring.is_cuda() && value_ring.is_cuda() &&
        query_local.scalar_type() == torch::kFloat32 && key_ring.scalar_type() == torch::kFloat32 &&
        value_ring.scalar_type() == torch::kFloat32) {
      auto out_lse = cp_ring_attention_cuda_forward_with_lse(
          query_local, key_ring, value_ring, positions, query_begin, original_sequence_length, shard_size, scale);
      ctx->save_for_backward({query_local, key_ring, value_ring, key_begin_positions, out_lse.at(1)});
      return out_lse.at(0);
    }
    ctx->save_for_backward({query_local, key_ring, value_ring, key_begin_positions});
    return streaming_attention_from_ring_blocks(query_local,
                                                key_ring,
                                                value_ring,
                                                positions,
                                                query_begin,
                                                original_sequence_length,
                                                shard_size,
                                                scale);
  }

  static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx,
                                               torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    const int64_t query_begin = ctx->saved_data["query_begin"].toInt();
    const int64_t original_sequence_length = ctx->saved_data["original_sequence_length"].toInt();
    const int64_t shard_size = ctx->saved_data["shard_size"].toInt();
    const double scale = ctx->saved_data["scale"].toDouble();
    const auto key_begin_positions = vector_from_cpu_i64_tensor(saved.at(3));

    if (cp_attention_cuda_available() && saved.at(0).is_cuda() && saved.at(1).is_cuda() && saved.at(2).is_cuda() &&
        saved.at(0).scalar_type() == torch::kFloat32 && saved.at(1).scalar_type() == torch::kFloat32 &&
        saved.at(2).scalar_type() == torch::kFloat32 && grad_outputs.at(0).is_cuda() &&
        grad_outputs.at(0).scalar_type() == torch::kFloat32) {
      auto grads = saved.size() > 4 && saved.at(4).defined()
                       ? cp_ring_attention_cuda_backward_with_lse(grad_outputs.at(0).contiguous(),
                                                                  saved.at(0).contiguous(),
                                                                  saved.at(1).contiguous(),
                                                                  saved.at(2).contiguous(),
                                                                  saved.at(4).contiguous(),
                                                                  key_begin_positions,
                                                                  query_begin,
                                                                  original_sequence_length,
                                                                  shard_size,
                                                                  scale)
                       : cp_ring_attention_cuda_backward(grad_outputs.at(0).contiguous(),
                                                         saved.at(0).contiguous(),
                                                         saved.at(1).contiguous(),
                                                         saved.at(2).contiguous(),
                                                         key_begin_positions,
                                                         query_begin,
                                                         original_sequence_length,
                                                         shard_size,
                                                         scale);
      return {grads.at(0), grads.at(1), grads.at(2), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
              torch::Tensor()};
    }

    torch::AutoGradMode enable_grad(true);
    auto query = saved.at(0).detach().set_requires_grad(true);
    auto key = saved.at(1).detach().set_requires_grad(true);
    auto value = saved.at(2).detach().set_requires_grad(true);
    auto out = streaming_attention_from_ring_blocks(
        query, key, value, key_begin_positions, query_begin, original_sequence_length, shard_size, scale);
    auto grads = torch::autograd::grad(
        {out}, {query, key, value}, {grad_outputs.at(0).contiguous()}, false, false, true);
    return {grads.at(0), grads.at(1), grads.at(2), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor()};
  }
};

}  // namespace

int64_t context_parallel_padded_length(int64_t sequence_length, int64_t context_parallel) {
  if (sequence_length < 0) {
    throw std::invalid_argument("sequence_length must be non-negative");
  }
  if (context_parallel <= 0) {
    throw std::invalid_argument("context_parallel must be positive");
  }
  const int64_t remainder = sequence_length % context_parallel;
  return remainder == 0 ? sequence_length : sequence_length + context_parallel - remainder;
}

ContextShardRange context_parallel_sequence_range(int64_t sequence_length,
                                                  int64_t context_rank,
                                                  int64_t context_parallel) {
  validate_context_rank(context_rank, context_parallel);
  const int64_t padded = context_parallel_padded_length(sequence_length, context_parallel);
  const int64_t shard = padded / context_parallel;
  const int64_t padded_begin = context_rank * shard;
  const int64_t padded_end = padded_begin + shard;
  return ContextShardRange{
      std::min(padded_begin, sequence_length),
      std::min(padded_end, sequence_length),
      padded_end,
  };
}

torch::Tensor context_parallel_slice(const torch::Tensor& input,
                                     int64_t context_rank,
                                     int64_t context_parallel,
                                     int64_t sequence_dim) {
  validate_context_rank(context_rank, context_parallel);
  const int64_t dim = normalize_sequence_dim(input, sequence_dim);
  const int64_t seq = input.size(dim);
  if (seq % context_parallel != 0) {
    throw std::invalid_argument("sequence length must be divisible by context_parallel");
  }
  const int64_t shard = seq / context_parallel;
  return input.narrow(dim, context_rank * shard, shard).contiguous();
}

torch::Tensor context_parallel_slice_padded(const torch::Tensor& input,
                                            int64_t context_rank,
                                            int64_t context_parallel,
                                            int64_t sequence_dim,
                                            double pad_value) {
  validate_context_rank(context_rank, context_parallel);
  const int64_t dim = normalize_sequence_dim(input, sequence_dim);
  const int64_t seq = input.size(dim);
  auto range = context_parallel_sequence_range(seq, context_rank, context_parallel);
  const int64_t padded = context_parallel_padded_length(seq, context_parallel);
  const int64_t padded_shard = padded / context_parallel;
  auto sizes = input.sizes().vec();
  sizes[static_cast<size_t>(dim)] = padded_shard;
  auto out = torch::full(sizes, pad_value, input.options());
  const int64_t valid = range.end - range.begin;
  if (valid > 0) {
    out.narrow(dim, 0, valid).copy_(input.narrow(dim, range.begin, valid));
  }
  return out.contiguous();
}

torch::Tensor context_parallel_gather(const torch::Tensor& local,
                                      Collectives& collectives,
                                      const std::vector<int64_t>& context_group,
                                      int64_t sequence_dim) {
  if (context_group.size() <= 1) {
    return local;
  }
  const int64_t dim = normalize_sequence_dim(local, sequence_dim);
  if (dim == 0) {
    return collectives.all_gather(local.contiguous(), context_group, 0);
  }
  auto moved = local.transpose(0, dim).contiguous();
  auto gathered = collectives.all_gather(moved, context_group, 0);
  return gathered.transpose(0, dim).contiguous();
}

torch::Tensor context_parallel_gather_autograd(const torch::Tensor& local,
                                               Collectives& collectives,
                                               const std::vector<int64_t>& context_group,
                                               int64_t sequence_dim) {
  if (context_group.size() <= 1) {
    return local;
  }
  const int64_t dim = normalize_sequence_dim(local, sequence_dim);
  auto group_tensor = group_tensor_from_vector(context_group);
  const auto collectives_addr = static_cast<int64_t>(reinterpret_cast<uintptr_t>(&collectives));
  return ContextParallelGatherFunction::apply(local, group_tensor, collectives_addr, dim);
}

torch::Tensor context_parallel_ring_exchange_autograd(const torch::Tensor& local,
                                                      Collectives& collectives,
                                                      const std::vector<int64_t>& context_group,
                                                      int64_t context_rank,
                                                      int64_t sequence_dim) {
  validate_context_rank(context_rank, static_cast<int64_t>(context_group.size()));
  if (context_group.size() <= 1) {
    return local;
  }
  const int64_t dim = normalize_sequence_dim(local, sequence_dim);
  auto group_tensor = group_tensor_from_vector(context_group);
  const auto collectives_addr = static_cast<int64_t>(reinterpret_cast<uintptr_t>(&collectives));
  return ContextParallelRingExchangeFunction::apply(local, group_tensor, collectives_addr, context_rank, dim);
}

torch::Tensor context_parallel_gather_padded(const torch::Tensor& local,
                                             Collectives& collectives,
                                             const std::vector<int64_t>& context_group,
                                             int64_t sequence_dim,
                                             int64_t original_sequence_length) {
  if (original_sequence_length < 0) {
    throw std::invalid_argument("original_sequence_length must be non-negative");
  }
  auto gathered = context_parallel_gather(local, collectives, context_group, sequence_dim);
  const int64_t dim = normalize_sequence_dim(gathered, sequence_dim);
  if (gathered.size(dim) < original_sequence_length) {
    throw std::invalid_argument("gathered context tensor is shorter than original_sequence_length");
  }
  return gathered.narrow(dim, 0, original_sequence_length).contiguous();
}

int64_t context_parallel_group_index(const std::vector<int64_t>& context_group, int64_t rank) {
  for (size_t i = 0; i < context_group.size(); ++i) {
    if (context_group[i] == rank) {
      return static_cast<int64_t>(i);
    }
  }
  throw std::invalid_argument("rank is not a member of context_group");
}

std::vector<ContextParallelRingStep> context_parallel_ring_schedule(const std::vector<int64_t>& context_group,
                                                                    int64_t rank) {
  if (context_group.empty()) {
    throw std::invalid_argument("context_group must not be empty");
  }
  const int64_t cp = static_cast<int64_t>(context_group.size());
  const int64_t local = context_parallel_group_index(context_group, rank);
  const int64_t send_rank = cp == 1 ? -1 : context_group[static_cast<size_t>((local + 1) % cp)];
  const int64_t recv_rank = cp == 1 ? -1 : context_group[static_cast<size_t>((local + cp - 1) % cp)];
  std::vector<ContextParallelRingStep> schedule;
  schedule.reserve(context_group.size());
  for (int64_t step = 0; step < cp; ++step) {
    const int64_t kv_index = (local + cp - step) % cp;
    schedule.push_back(ContextParallelRingStep{step, context_group[static_cast<size_t>(kv_index)], send_rank, recv_rank});
  }
  return schedule;
}

torch::Tensor context_parallel_causal_attention(const torch::Tensor& query_local,
                                                const torch::Tensor& key_global,
                                                const torch::Tensor& value_global,
                                                int64_t query_begin,
                                                double scale) {
  require_attention_tensor(query_local, "query_local");
  require_attention_tensor(key_global, "key_global");
  require_attention_tensor(value_global, "value_global");
  if (query_begin < 0) {
    throw std::invalid_argument("query_begin must be non-negative");
  }
  if (query_local.size(0) != key_global.size(0) || query_local.size(0) != value_global.size(0) ||
      query_local.size(1) != key_global.size(1) || query_local.size(1) != value_global.size(1) ||
      key_global.size(2) != value_global.size(2) || query_local.size(3) != key_global.size(3)) {
    throw std::invalid_argument("context causal attention shape mismatch");
  }
  const int64_t local_seq = query_local.size(2);
  const int64_t global_seq = key_global.size(2);
  if (query_begin + local_seq > global_seq) {
    throw std::invalid_argument("query shard range exceeds global KV sequence length");
  }

  auto scores = torch::matmul(query_local.to(torch::kFloat32), key_global.to(torch::kFloat32).transpose(2, 3)) * scale;
  auto q_pos = torch::arange(query_begin,
                             query_begin + local_seq,
                             torch::TensorOptions().dtype(torch::kLong).device(query_local.device()))
                   .view({local_seq, 1});
  auto k_pos =
      torch::arange(0, global_seq, torch::TensorOptions().dtype(torch::kLong).device(query_local.device()))
          .view({1, global_seq});
  auto mask = k_pos > q_pos;
  scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1.0e9);
  auto probs = torch::softmax(scores, -1);
  return torch::matmul(probs, value_global.to(torch::kFloat32)).to(query_local.scalar_type());
}

torch::Tensor context_parallel_causal_attention_streaming_blocks(
    const torch::Tensor& query_local,
    const std::vector<torch::Tensor>& key_blocks,
    const std::vector<torch::Tensor>& value_blocks,
    const std::vector<int64_t>& key_begin_positions,
    int64_t query_begin,
    int64_t original_sequence_length,
    double scale) {
  require_attention_tensor(query_local, "query_local");
  if (query_begin < 0 || original_sequence_length < 0) {
    throw std::invalid_argument("query_begin and original_sequence_length must be non-negative");
  }
  if (key_blocks.size() != value_blocks.size() || key_blocks.size() != key_begin_positions.size()) {
    throw std::invalid_argument("streaming CP attention block metadata size mismatch");
  }
  if (key_blocks.empty()) {
    throw std::invalid_argument("streaming CP attention requires at least one KV block");
  }
  const int64_t local_seq = query_local.size(2);
  const int64_t valid_query =
      std::max<int64_t>(0, std::min<int64_t>(local_seq, original_sequence_length - query_begin));
  auto out = torch::zeros({query_local.size(0), query_local.size(1), local_seq, value_blocks.front().size(3)},
                          query_local.options());
  if (valid_query == 0) {
    return out;
  }

  auto query = query_local.narrow(2, 0, valid_query).contiguous().to(torch::kFloat32);
  auto q_pos = torch::arange(query_begin,
                             query_begin + valid_query,
                             torch::TensorOptions().dtype(torch::kLong).device(query_local.device()))
                   .view({valid_query, 1});
  const double neg_inf = -std::numeric_limits<float>::infinity();
  auto row_max = torch::full({query.size(0), query.size(1), valid_query, 1},
                             neg_inf,
                             torch::TensorOptions().dtype(torch::kFloat32).device(query_local.device()));
  auto row_sum = torch::zeros_like(row_max);
  auto acc = torch::zeros({query.size(0), query.size(1), valid_query, value_blocks.front().size(3)},
                          torch::TensorOptions().dtype(torch::kFloat32).device(query_local.device()));

  for (size_t i = 0; i < key_blocks.size(); ++i) {
    const auto& key_block = key_blocks[i];
    const auto& value_block = value_blocks[i];
    require_attention_tensor(key_block, "key_block");
    require_attention_tensor(value_block, "value_block");
    if (key_block.size(0) != query_local.size(0) || value_block.size(0) != query_local.size(0) ||
        key_block.size(1) != query_local.size(1) || value_block.size(1) != query_local.size(1) ||
        key_block.size(2) != value_block.size(2) || key_block.size(3) != query_local.size(3)) {
      throw std::invalid_argument("streaming CP attention block shape mismatch");
    }
    if (key_begin_positions[i] < 0) {
      throw std::invalid_argument("streaming CP attention key block position must be non-negative");
    }
    const int64_t valid_key =
        std::max<int64_t>(0, std::min<int64_t>(key_block.size(2), original_sequence_length - key_begin_positions[i]));
    if (valid_key == 0) {
      continue;
    }
    auto key = key_block.narrow(2, 0, valid_key).contiguous().to(torch::kFloat32);
    auto value = value_block.narrow(2, 0, valid_key).contiguous().to(torch::kFloat32);
    auto scores = torch::matmul(query, key.transpose(2, 3)) * scale;
    auto k_pos = torch::arange(key_begin_positions[i],
                               key_begin_positions[i] + valid_key,
                               torch::TensorOptions().dtype(torch::kLong).device(query_local.device()))
                     .view({1, valid_key});
    const auto valid_mask = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0);
    scores = scores.masked_fill(torch::logical_not(valid_mask), neg_inf);

    auto block_max = std::get<0>(scores.max(-1, true));
    auto next_max = torch::maximum(row_max, block_max);
    auto has_any = torch::isfinite(next_max);
    auto safe_next_max = torch::where(has_any, next_max, torch::zeros_like(next_max));
    auto old_scale = torch::where(torch::isfinite(row_max), torch::exp(row_max - safe_next_max), torch::zeros_like(row_max));
    auto exp_scores =
        torch::where(valid_mask, torch::exp(scores - safe_next_max), torch::zeros_like(scores));
    acc = acc * old_scale + torch::matmul(exp_scores, value);
    row_sum = row_sum * old_scale + exp_scores.sum(-1, true);
    row_max = next_max;
  }

  auto valid_out = (acc / row_sum.clamp_min(1.0e-20)).to(query_local.scalar_type());
  out.narrow(2, 0, valid_query).copy_(valid_out);
  return out;
}

torch::Tensor context_parallel_causal_attention_ring_blocks_recompute(
    const torch::Tensor& query_local,
    const torch::Tensor& key_ring,
    const torch::Tensor& value_ring,
    const std::vector<int64_t>& key_begin_positions,
    int64_t query_begin,
    int64_t original_sequence_length,
    int64_t shard_size,
    double scale) {
  if (key_begin_positions.empty()) {
    throw std::invalid_argument("CP recompute ring attention requires at least one KV block");
  }
  auto positions = vector_to_cpu_i64_tensor(key_begin_positions);
  return ContextParallelRingAttentionRecomputeFunction::apply(query_local.contiguous(),
                                                              key_ring.contiguous(),
                                                              value_ring.contiguous(),
                                                              positions,
                                                              query_begin,
                                                              original_sequence_length,
                                                              shard_size,
                                                              scale);
}

torch::Tensor context_parallel_causal_attention_gather_kv(const torch::Tensor& query_local,
                                                          const torch::Tensor& key_local,
                                                          const torch::Tensor& value_local,
                                                          Collectives& collectives,
                                                          const std::vector<int64_t>& context_group,
                                                          int64_t context_rank,
                                                          int64_t original_sequence_length,
                                                          double scale) {
  require_attention_tensor(query_local, "query_local");
  require_attention_tensor(key_local, "key_local");
  require_attention_tensor(value_local, "value_local");
  validate_context_rank(context_rank, static_cast<int64_t>(context_group.size()));
  const int64_t shard = query_local.size(2);
  if (key_local.size(2) != shard || value_local.size(2) != shard) {
    throw std::invalid_argument("query/key/value local CP shards must have matching sequence length");
  }
  auto key_global = context_parallel_gather_autograd(key_local.contiguous(), collectives, context_group, 2).contiguous();
  auto value_global =
      context_parallel_gather_autograd(value_local.contiguous(), collectives, context_group, 2).contiguous();
  if (key_global.size(2) < original_sequence_length || value_global.size(2) < original_sequence_length) {
    throw std::invalid_argument("gathered CP KV is shorter than original_sequence_length");
  }
  std::vector<torch::Tensor> key_blocks;
  std::vector<torch::Tensor> value_blocks;
  std::vector<int64_t> key_begin_positions;
  key_blocks.reserve(context_group.size());
  value_blocks.reserve(context_group.size());
  key_begin_positions.reserve(context_group.size());
  for (int64_t rank_index = 0; rank_index < static_cast<int64_t>(context_group.size()); ++rank_index) {
    const int64_t begin = rank_index * shard;
    if (begin >= key_global.size(2)) {
      break;
    }
    const int64_t block = std::min<int64_t>(shard, key_global.size(2) - begin);
    key_blocks.push_back(key_global.narrow(2, begin, block).contiguous());
    value_blocks.push_back(value_global.narrow(2, begin, block).contiguous());
    key_begin_positions.push_back(begin);
  }
  return context_parallel_causal_attention_streaming_blocks(
      query_local, key_blocks, value_blocks, key_begin_positions, context_rank * shard, original_sequence_length, scale);
}

torch::Tensor context_parallel_causal_attention_ring_gather_kv(const torch::Tensor& query_local,
                                                               const torch::Tensor& key_local,
                                                               const torch::Tensor& value_local,
                                                               Collectives& collectives,
                                                               const std::vector<int64_t>& context_group,
                                                               int64_t context_rank,
                                                               int64_t original_sequence_length,
                                                               double scale) {
  require_attention_tensor(query_local, "query_local");
  require_attention_tensor(key_local, "key_local");
  require_attention_tensor(value_local, "value_local");
  validate_context_rank(context_rank, static_cast<int64_t>(context_group.size()));
  const int64_t shard = query_local.size(2);
  if (key_local.size(2) != shard || value_local.size(2) != shard) {
    throw std::invalid_argument("query/key/value local CP shards must have matching sequence length");
  }
  auto key_global = context_parallel_gather_autograd(key_local.contiguous(), collectives, context_group, 2).contiguous();
  auto value_global =
      context_parallel_gather_autograd(value_local.contiguous(), collectives, context_group, 2).contiguous();
  if (key_global.size(2) < original_sequence_length || value_global.size(2) < original_sequence_length) {
    throw std::invalid_argument("gathered CP KV is shorter than original_sequence_length");
  }

  std::vector<torch::Tensor> key_blocks;
  std::vector<torch::Tensor> value_blocks;
  std::vector<int64_t> key_begin_positions;
  const auto schedule = context_parallel_ring_schedule(context_group, context_group.at(static_cast<size_t>(context_rank)));
  key_blocks.reserve(schedule.size());
  value_blocks.reserve(schedule.size());
  key_begin_positions.reserve(schedule.size());
  for (const auto& step : schedule) {
    const int64_t rank_index = context_parallel_group_index(context_group, step.kv_rank);
    const int64_t begin = rank_index * shard;
    if (begin >= key_global.size(2)) {
      continue;
    }
    const int64_t block = std::min<int64_t>(shard, key_global.size(2) - begin);
    key_blocks.push_back(key_global.narrow(2, begin, block).contiguous());
    value_blocks.push_back(value_global.narrow(2, begin, block).contiguous());
    key_begin_positions.push_back(begin);
  }
  return context_parallel_causal_attention_streaming_blocks(
      query_local, key_blocks, value_blocks, key_begin_positions, context_rank * shard, original_sequence_length, scale);
}

torch::Tensor context_parallel_causal_attention_ring_exchange_kv(const torch::Tensor& query_local,
                                                                 const torch::Tensor& key_local,
                                                                 const torch::Tensor& value_local,
                                                                 Collectives& collectives,
                                                                 const std::vector<int64_t>& context_group,
                                                                 int64_t context_rank,
                                                                 int64_t original_sequence_length,
                                                                 double scale) {
  require_attention_tensor(query_local, "query_local");
  require_attention_tensor(key_local, "key_local");
  require_attention_tensor(value_local, "value_local");
  validate_context_rank(context_rank, static_cast<int64_t>(context_group.size()));
  const int64_t shard = query_local.size(2);
  if (key_local.size(2) != shard || value_local.size(2) != shard) {
    throw std::invalid_argument("query/key/value local CP shards must have matching sequence length");
  }
  if (key_local.sizes().slice(0, 3) != value_local.sizes().slice(0, 3)) {
    throw std::invalid_argument("key/value local CP shards must match [B,H,S] for fused ring exchange");
  }
  auto kv_local = torch::cat({key_local.contiguous(), value_local.contiguous()}, -1).contiguous();
  auto kv_ring = context_parallel_ring_exchange_autograd(kv_local, collectives, context_group, context_rank, 2)
                     .contiguous();
  auto key_ring = kv_ring.narrow(-1, 0, key_local.size(-1)).contiguous();
  auto value_ring = kv_ring.narrow(-1, key_local.size(-1), value_local.size(-1)).contiguous();
  const auto schedule = context_parallel_ring_schedule(context_group, context_group.at(static_cast<size_t>(context_rank)));
  std::vector<int64_t> key_begin_positions;
  key_begin_positions.reserve(schedule.size());
  for (size_t i = 0; i < schedule.size(); ++i) {
    const int64_t rank_index = context_parallel_group_index(context_group, schedule[i].kv_rank);
    key_begin_positions.push_back(rank_index * shard);
  }
  return context_parallel_causal_attention_ring_blocks_recompute(
      query_local, key_ring, value_ring, key_begin_positions, context_rank * shard, original_sequence_length, shard, scale);
}

}  // namespace cverl::distributed
