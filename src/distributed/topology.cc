#include "cverl/distributed/topology.h"

#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace cverl::distributed {
namespace {

int64_t env_i64(const char* name, int64_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  return std::stoll(value);
}

std::string join(const std::vector<std::string>& values, const char* sep) {
  std::ostringstream out;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << sep;
    }
    out << values[i];
  }
  return out.str();
}

void require_positive(int64_t value, const char* name) {
  if (value <= 0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
}

}  // namespace

std::string dtype_policy_name(DTypePolicy dtype) {
  switch (dtype) {
    case DTypePolicy::Float32:
      return "float32";
    case DTypePolicy::Float16:
      return "float16";
    case DTypePolicy::BFloat16:
      return "bfloat16";
  }
  throw std::invalid_argument("unknown dtype policy");
}

Topology::Topology(ClusterSpec spec) : spec_(std::move(spec)) {
  validate();
}

void Topology::validate() const {
  require_positive(spec_.world_size, "world_size");
  require_positive(spec_.local_world_size, "local_world_size");
  require_positive(spec_.gpus_per_node, "gpus_per_node");
  require_positive(spec_.parallel.data_parallel, "data_parallel");
  require_positive(spec_.parallel.tensor_parallel, "tensor_parallel");
  require_positive(spec_.parallel.pipeline_parallel, "pipeline_parallel");
  require_positive(spec_.parallel.micro_batches, "micro_batches");

  if (spec_.rank < 0 || spec_.rank >= spec_.world_size) {
    throw std::invalid_argument("rank must be in [0, world_size)");
  }
  if (spec_.local_rank < 0 || spec_.local_rank >= spec_.local_world_size) {
    throw std::invalid_argument("local_rank must be in [0, local_world_size)");
  }
  const int64_t expected_world =
      spec_.parallel.data_parallel * spec_.parallel.tensor_parallel * spec_.parallel.pipeline_parallel;
  if (spec_.world_size != expected_world) {
    throw std::invalid_argument("world_size must equal data_parallel * tensor_parallel * pipeline_parallel");
  }
  if (spec_.parallel.pipeline_parallel > 1 && spec_.parallel.micro_batches < spec_.parallel.pipeline_parallel) {
    throw std::invalid_argument("pipeline parallelism needs at least as many micro_batches as pipeline stages");
  }
  if (spec_.parallel.tensor_parallel > spec_.gpus_per_node) {
    throw std::invalid_argument("tensor_parallel should fit within one node for high-bandwidth TP collectives");
  }
  if (spec_.local_world_size > spec_.gpus_per_node) {
    throw std::invalid_argument("local_world_size cannot exceed gpus_per_node");
  }
}

int64_t Topology::global_rank(int64_t data_rank, int64_t pipeline_rank, int64_t tensor_rank) const {
  const auto& p = spec_.parallel;
  if (data_rank < 0 || data_rank >= p.data_parallel || pipeline_rank < 0 || pipeline_rank >= p.pipeline_parallel ||
      tensor_rank < 0 || tensor_rank >= p.tensor_parallel) {
    throw std::out_of_range("parallel rank coordinate out of range");
  }
  return ((data_rank * p.pipeline_parallel) + pipeline_rank) * p.tensor_parallel + tensor_rank;
}

ParallelRankInfo Topology::rank_info(int64_t rank) const {
  if (rank < 0 || rank >= spec_.world_size) {
    throw std::out_of_range("rank out of range");
  }
  const auto& p = spec_.parallel;
  ParallelRankInfo info;
  info.rank = rank;
  info.tensor_rank = rank % p.tensor_parallel;
  int64_t rem = rank / p.tensor_parallel;
  info.pipeline_rank = rem % p.pipeline_parallel;
  info.data_rank = rem / p.pipeline_parallel;

  for (int64_t d = 0; d < p.data_parallel; ++d) {
    info.data_group.push_back(global_rank(d, info.pipeline_rank, info.tensor_rank));
  }
  for (int64_t t = 0; t < p.tensor_parallel; ++t) {
    info.tensor_group.push_back(global_rank(info.data_rank, info.pipeline_rank, t));
  }
  for (int64_t pp = 0; pp < p.pipeline_parallel; ++pp) {
    info.pipeline_group.push_back(global_rank(info.data_rank, pp, info.tensor_rank));
  }
  return info;
}

LayerRange Topology::pipeline_layer_range(int64_t num_layers, int64_t pipeline_rank) const {
  if (num_layers < 0) {
    throw std::invalid_argument("num_layers must be non-negative");
  }
  const int64_t pp = spec_.parallel.pipeline_parallel;
  if (pipeline_rank < 0 || pipeline_rank >= pp) {
    throw std::out_of_range("pipeline rank out of range");
  }
  int64_t base = num_layers / pp;
  int64_t rem = num_layers % pp;
  int64_t begin = pipeline_rank * base + std::min(pipeline_rank, rem);
  int64_t count = base + (pipeline_rank < rem ? 1 : 0);
  return LayerRange{begin, begin + count};
}

LayerRange Topology::local_pipeline_layer_range(int64_t num_layers) const {
  return pipeline_layer_range(num_layers, local_rank_info().pipeline_rank);
}

std::map<std::string, std::string> Topology::nccl_env() const {
  std::map<std::string, std::string> env;
  env["NCCL_P2P_DISABLE"] = spec_.network.enable_p2p ? "0" : "1";
  env["NCCL_IB_DISABLE"] = spec_.network.enable_ib ? "0" : "1";
  env["NCCL_NVLS_ENABLE"] = spec_.network.enable_nvls ? "1" : "0";
  env["NCCL_ASYNC_ERROR_HANDLING"] = "1";
  env["TORCH_NCCL_BLOCKING_WAIT"] = "0";
  if (!spec_.network.socket_ifnames.empty()) {
    env["NCCL_SOCKET_IFNAME"] = join(spec_.network.socket_ifnames, ",");
  }
  if (!spec_.network.ib_hcas.empty()) {
    env["NCCL_IB_HCA"] = join(spec_.network.ib_hcas, ",");
  }
  if (!spec_.network.nccl_topo_file.empty()) {
    env["NCCL_TOPO_FILE"] = spec_.network.nccl_topo_file;
  }
  if (spec_.network.nccl_ib_gid_index >= 0) {
    env["NCCL_IB_GID_INDEX"] = std::to_string(spec_.network.nccl_ib_gid_index);
  }
  return env;
}

ClusterSpec cluster_spec_from_env(const ParallelDims& parallel, MemoryPolicy memory, NetworkPolicy network) {
  ClusterSpec spec;
  spec.world_size = env_i64("WORLD_SIZE", 1);
  spec.rank = env_i64("RANK", 0);
  spec.local_world_size = env_i64("LOCAL_WORLD_SIZE", spec.world_size);
  spec.local_rank = env_i64("LOCAL_RANK", spec.rank);
  spec.node_rank = env_i64("NODE_RANK", spec.local_world_size > 0 ? spec.rank / spec.local_world_size : 0);
  spec.gpus_per_node = env_i64("CVERL_GPUS_PER_NODE", spec.local_world_size);
  spec.parallel = parallel;
  spec.memory = memory;
  spec.network = std::move(network);
  return spec;
}

}  // namespace cverl::distributed
