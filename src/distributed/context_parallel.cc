#include "cverl/distributed/context_parallel.h"

#include <algorithm>
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
  auto key_global = collectives.all_gather(key_local.contiguous(), context_group, 2).contiguous();
  auto value_global = collectives.all_gather(value_local.contiguous(), context_group, 2).contiguous();
  if (key_global.size(2) < original_sequence_length || value_global.size(2) < original_sequence_length) {
    throw std::invalid_argument("gathered CP KV is shorter than original_sequence_length");
  }
  key_global = key_global.narrow(2, 0, original_sequence_length).contiguous();
  value_global = value_global.narrow(2, 0, original_sequence_length).contiguous();
  const int64_t query_begin = context_rank * shard;
  return context_parallel_causal_attention(query_local, key_global, value_global, query_begin, scale);
}

}  // namespace cverl::distributed
