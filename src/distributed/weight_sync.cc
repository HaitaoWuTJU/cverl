#include "cverl/distributed/weight_sync.h"

#ifdef CVERL_ENABLE_NCCL
#include "cverl/distributed/nccl_collectives.h"
#endif

#include <stdexcept>
#include <vector>

namespace cverl::distributed {

namespace {

void require_parameter(const ParameterView& p) {
  if (p.name.empty()) {
    throw std::invalid_argument("parameter name must not be empty");
  }
  if (!p.tensor.defined()) {
    throw std::invalid_argument("undefined parameter: " + p.name);
  }
  if (!p.tensor.is_contiguous()) {
    throw std::invalid_argument("parameter must be contiguous: " + p.name);
  }
}

int64_t tensor_bytes(const torch::Tensor& tensor) {
  return tensor.numel() * static_cast<int64_t>(tensor.element_size());
}

void require_rank_in_group(int64_t rank,
                           int64_t root,
                           const std::vector<int64_t>& group,
                           const char* name) {
  if (group.empty()) {
    throw std::invalid_argument(std::string(name) + " requires a non-empty group");
  }
  bool have_rank = false;
  bool have_root = false;
  for (int64_t member : group) {
    have_rank = have_rank || member == rank;
    have_root = have_root || member == root;
  }
  if (!have_rank) {
    throw std::invalid_argument(std::string(name) + " collectives rank is not in group");
  }
  if (!have_root) {
    throw std::invalid_argument(std::string(name) + " root is not in group");
  }
}

struct ParameterBucketEntry {
  const ParameterView* parameter = nullptr;
  int64_t numel = 0;
};

void flush_broadcast_bucket(std::vector<ParameterBucketEntry>* bucket,
                            Collectives& collectives,
                            int64_t root,
                            const std::vector<int64_t>& group) {
  if (bucket->empty()) {
    return;
  }
  std::vector<torch::Tensor> flat;
  flat.reserve(bucket->size());
  for (const auto& entry : *bucket) {
    flat.push_back(entry.parameter->tensor.contiguous().view({entry.numel}));
  }
  auto payload = torch::cat(flat, 0).contiguous();
  auto synced = collectives.broadcast(payload, root, group).contiguous();
  if (synced.numel() != payload.numel() || synced.scalar_type() != payload.scalar_type()) {
    throw std::runtime_error("broadcast returned incompatible parameter bucket");
  }
  int64_t offset = 0;
  for (const auto& entry : *bucket) {
    auto& tensor = entry.parameter->tensor;
    auto shard = synced.narrow(0, offset, entry.numel).view(tensor.sizes());
    tensor.copy_(shard);
    offset += entry.numel;
  }
  bucket->clear();
}

#ifdef CVERL_ENABLE_NCCL
void flush_send_bucket(std::vector<ParameterBucketEntry>* bucket,
                       NcclCollectives& collectives,
                       int64_t peer) {
  if (bucket->empty()) {
    return;
  }
  std::vector<torch::Tensor> flat;
  flat.reserve(bucket->size());
  for (const auto& entry : *bucket) {
    flat.push_back(entry.parameter->tensor.contiguous().view({entry.numel}));
  }
  collectives.send(torch::cat(flat, 0).contiguous(), peer);
  bucket->clear();
}

void flush_recv_bucket(std::vector<ParameterBucketEntry>* bucket,
                       NcclCollectives& collectives,
                       int64_t peer) {
  if (bucket->empty()) {
    return;
  }
  int64_t total_numel = 0;
  for (const auto& entry : *bucket) {
    total_numel += entry.numel;
  }
  const auto& first = bucket->front().parameter->tensor;
  auto like = torch::empty({total_numel}, first.options());
  auto received = collectives.recv_like(like.contiguous(), peer).contiguous();
  if (received.numel() != total_numel || received.scalar_type() != first.scalar_type()) {
    throw std::runtime_error("received incompatible parameter shard bucket");
  }
  int64_t offset = 0;
  for (const auto& entry : *bucket) {
    auto& tensor = entry.parameter->tensor;
    auto shard = received.narrow(0, offset, entry.numel).view(tensor.sizes());
    tensor.copy_(shard);
    offset += entry.numel;
  }
  bucket->clear();
}
#endif

}  // namespace

void broadcast_parameters_from_root(const std::vector<ParameterView>& parameters,
                                    Collectives& collectives,
                                    int64_t root,
                                    const std::vector<int64_t>& group,
                                    int64_t bucket_bytes,
                                    bool final_barrier) {
  if (parameters.empty()) {
    return;
  }
  if (bucket_bytes <= 0) {
    throw std::invalid_argument("broadcast_parameters_from_root bucket_bytes must be positive");
  }
  require_rank_in_group(collectives.rank(), root, group, "broadcast_parameters_from_root");
  torch::NoGradGuard no_grad;
  if (group.size() == 1) {
    for (const auto& p : parameters) {
      require_parameter(p);
    }
    return;
  }
  std::vector<ParameterBucketEntry> bucket;
  bucket.reserve(parameters.size());
  int64_t current_bytes = 0;
  torch::ScalarType current_dtype = torch::kFloat32;
  torch::Device current_device(torch::kCPU);
  bool have_bucket = false;
  for (const auto& p : parameters) {
    require_parameter(p);
    const int64_t param_bytes = tensor_bytes(p.tensor);
    const bool incompatible =
        have_bucket && (p.tensor.scalar_type() != current_dtype || p.tensor.device() != current_device);
    const bool would_overflow = have_bucket && current_bytes + param_bytes > bucket_bytes;
    if (incompatible || would_overflow) {
      flush_broadcast_bucket(&bucket, collectives, root, group);
      current_bytes = 0;
      have_bucket = false;
    }
    if (!have_bucket) {
      current_dtype = p.tensor.scalar_type();
      current_device = p.tensor.device();
      have_bucket = true;
    }
    bucket.push_back(ParameterBucketEntry{&p, p.tensor.numel()});
    current_bytes += param_bytes;
  }
  flush_broadcast_bucket(&bucket, collectives, root, group);
  if (final_barrier) {
    collectives.barrier();
  }
}

std::vector<ParameterView> module_parameter_views(torch::nn::Module& module, bool recurse) {
  std::vector<ParameterView> out;
  auto named = module.named_parameters(recurse);
  out.reserve(named.size());
  for (const auto& kv : named) {
    out.push_back(ParameterView{kv.key(), kv.value()});
  }
  return out;
}

#ifdef CVERL_ENABLE_NCCL
void send_parameter_shards(const std::vector<ParameterView>& shards,
                           NcclCollectives& collectives,
                           int64_t peer,
                           int64_t bucket_bytes) {
  if (bucket_bytes <= 0) {
    throw std::invalid_argument("send_parameter_shards bucket_bytes must be positive");
  }
  torch::NoGradGuard no_grad;
  std::vector<ParameterBucketEntry> bucket;
  bucket.reserve(shards.size());
  int64_t current_bytes = 0;
  torch::ScalarType current_dtype = torch::kFloat32;
  torch::Device current_device(torch::kCPU);
  bool have_bucket = false;
  for (const auto& shard : shards) {
    require_parameter(shard);
    const int64_t shard_bytes = tensor_bytes(shard.tensor);
    const bool incompatible =
        have_bucket && (shard.tensor.scalar_type() != current_dtype || shard.tensor.device() != current_device);
    const bool would_overflow = have_bucket && current_bytes + shard_bytes > bucket_bytes;
    if (incompatible || would_overflow) {
      flush_send_bucket(&bucket, collectives, peer);
      current_bytes = 0;
      have_bucket = false;
    }
    if (!have_bucket) {
      current_dtype = shard.tensor.scalar_type();
      current_device = shard.tensor.device();
      have_bucket = true;
    }
    bucket.push_back(ParameterBucketEntry{&shard, shard.tensor.numel()});
    current_bytes += shard_bytes;
  }
  flush_send_bucket(&bucket, collectives, peer);
}

void recv_parameter_shards(const std::vector<ParameterView>& shards,
                           NcclCollectives& collectives,
                           int64_t peer,
                           int64_t bucket_bytes) {
  if (bucket_bytes <= 0) {
    throw std::invalid_argument("recv_parameter_shards bucket_bytes must be positive");
  }
  torch::NoGradGuard no_grad;
  std::vector<ParameterBucketEntry> bucket;
  bucket.reserve(shards.size());
  int64_t current_bytes = 0;
  torch::ScalarType current_dtype = torch::kFloat32;
  torch::Device current_device(torch::kCPU);
  bool have_bucket = false;
  for (const auto& shard : shards) {
    require_parameter(shard);
    const int64_t shard_bytes = tensor_bytes(shard.tensor);
    const bool incompatible =
        have_bucket && (shard.tensor.scalar_type() != current_dtype || shard.tensor.device() != current_device);
    const bool would_overflow = have_bucket && current_bytes + shard_bytes > bucket_bytes;
    if (incompatible || would_overflow) {
      flush_recv_bucket(&bucket, collectives, peer);
      current_bytes = 0;
      have_bucket = false;
    }
    if (!have_bucket) {
      current_dtype = shard.tensor.scalar_type();
      current_device = shard.tensor.device();
      have_bucket = true;
    }
    bucket.push_back(ParameterBucketEntry{&shard, shard.tensor.numel()});
    current_bytes += shard_bytes;
  }
  flush_recv_bucket(&bucket, collectives, peer);
}
#endif

}  // namespace cverl::distributed
