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

struct ParameterBucketEntry {
  const ParameterView* parameter = nullptr;
  int64_t numel = 0;
};

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
                                    const std::vector<int64_t>& group) {
  if (parameters.empty()) {
    return;
  }
  torch::NoGradGuard no_grad;
  for (const auto& p : parameters) {
    require_parameter(p);
    auto synced = collectives.broadcast(p.tensor, root, group);
    if (synced.sizes() != p.tensor.sizes() || synced.scalar_type() != p.tensor.scalar_type()) {
      throw std::runtime_error("broadcast returned incompatible tensor for parameter: " + p.name);
    }
    p.tensor.copy_(synced);
  }
  collectives.barrier();
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
