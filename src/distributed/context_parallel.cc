#include "cverl/distributed/context_parallel.h"

#include <algorithm>
#include <stdexcept>

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

}  // namespace cverl::distributed
