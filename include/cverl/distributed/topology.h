#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace cverl::distributed {

enum class DTypePolicy {
  Float32,
  Float16,
  BFloat16,
};

struct ParallelDims {
  int64_t data_parallel = 1;
  int64_t tensor_parallel = 1;
  int64_t pipeline_parallel = 1;
  int64_t micro_batches = 1;
};

struct MemoryPolicy {
  DTypePolicy param_dtype = DTypePolicy::BFloat16;
  DTypePolicy reduce_dtype = DTypePolicy::Float32;
  bool activation_checkpointing = true;
  bool overlap_grad_reduce = true;
  bool overlap_param_gather = true;
  bool shard_optimizer = true;
  bool shard_gradients = true;
  bool cpu_optimizer_offload = false;
  bool cpu_param_offload = false;
  int64_t max_activation_bytes_per_rank = 0;
};

struct NetworkPolicy {
  std::vector<std::string> socket_ifnames;
  std::vector<std::string> ib_hcas;
  std::string nccl_topo_file;
  int64_t nccl_ib_gid_index = -1;
  bool enable_nvls = true;
  bool enable_p2p = true;
  bool enable_ib = true;
};

struct ClusterSpec {
  int64_t world_size = 1;
  int64_t rank = 0;
  int64_t local_world_size = 1;
  int64_t local_rank = 0;
  int64_t node_rank = 0;
  int64_t gpus_per_node = 1;
  ParallelDims parallel;
  MemoryPolicy memory;
  NetworkPolicy network;
};

struct ParallelRankInfo {
  int64_t rank = 0;
  int64_t data_rank = 0;
  int64_t tensor_rank = 0;
  int64_t pipeline_rank = 0;
  std::vector<int64_t> data_group;
  std::vector<int64_t> tensor_group;
  std::vector<int64_t> pipeline_group;
};

class Topology {
 public:
  explicit Topology(ClusterSpec spec);

  const ClusterSpec& spec() const { return spec_; }
  int64_t world_size() const { return spec_.world_size; }
  int64_t rank() const { return spec_.rank; }

  ParallelRankInfo rank_info(int64_t rank) const;
  ParallelRankInfo local_rank_info() const { return rank_info(spec_.rank); }
  int64_t global_rank(int64_t data_rank, int64_t pipeline_rank, int64_t tensor_rank) const;
  std::map<std::string, std::string> nccl_env() const;

  void validate() const;

 private:
  ClusterSpec spec_;
};

ClusterSpec cluster_spec_from_env(const ParallelDims& parallel,
                                  MemoryPolicy memory = {},
                                  NetworkPolicy network = {});

std::string dtype_policy_name(DTypePolicy dtype);

}  // namespace cverl::distributed
