#include "cverl/distributed/collectives.h"
#include "cverl/distributed/topology.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <torch/torch.h>

namespace {

void require(bool cond, const char* msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

void require_vec_eq(const std::vector<int64_t>& actual, const std::vector<int64_t>& expected, const char* msg) {
  if (actual != expected) {
    std::cerr << msg << "\nactual:";
    for (int64_t v : actual) {
      std::cerr << " " << v;
    }
    std::cerr << "\nexpected:";
    for (int64_t v : expected) {
      std::cerr << " " << v;
    }
    std::cerr << "\n";
    throw std::runtime_error(msg);
  }
}

template <typename Fn>
void require_throws(Fn&& fn, const char* msg) {
  bool failed = false;
  try {
    fn();
  } catch (const std::exception&) {
    failed = true;
  }
  require(failed, msg);
}

cverl::distributed::ClusterSpec make_spec() {
  cverl::distributed::ClusterSpec spec;
  spec.parallel.data_parallel = 2;
  spec.parallel.pipeline_parallel = 2;
  spec.parallel.tensor_parallel = 4;
  spec.parallel.micro_batches = 4;
  spec.world_size = 16;
  spec.rank = 6;
  spec.local_world_size = 8;
  spec.local_rank = 6;
  spec.gpus_per_node = 8;
  spec.network.socket_ifnames = {"eth0", "ib0"};
  spec.network.ib_hcas = {"mlx5_0", "mlx5_1"};
  spec.network.nccl_ib_gid_index = 3;
  return spec;
}

void test_rank_mapping() {
  cverl::distributed::Topology topology(make_spec());
  auto info = topology.local_rank_info();
  require(info.data_rank == 0, "data rank");
  require(info.pipeline_rank == 1, "pipeline rank");
  require(info.tensor_rank == 2, "tensor rank");
  require_vec_eq(info.tensor_group, {4, 5, 6, 7}, "tensor group");
  require_vec_eq(info.pipeline_group, {2, 6}, "pipeline group");
  require_vec_eq(info.data_group, {6, 14}, "data group");
  require(topology.global_rank(1, 1, 2) == 14, "global rank");
}

void test_nccl_env() {
  cverl::distributed::Topology topology(make_spec());
  auto env = topology.nccl_env();
  require(env.at("NCCL_SOCKET_IFNAME") == "eth0,ib0", "socket ifnames");
  require(env.at("NCCL_IB_HCA") == "mlx5_0,mlx5_1", "ib hcas");
  require(env.at("NCCL_IB_GID_INDEX") == "3", "gid index");
  require(env.at("NCCL_NVLS_ENABLE") == "1", "nvls");
  require(env.at("NCCL_ASYNC_ERROR_HANDLING") == "1", "async error handling");

  auto spec = make_spec();
  spec.network.enable_ib = false;
  spec.network.enable_p2p = false;
  spec.network.enable_nvls = false;
  spec.network.nccl_topo_file = "/tmp/topo.xml";
  cverl::distributed::Topology disabled(spec);
  env = disabled.nccl_env();
  require(env.at("NCCL_IB_DISABLE") == "1", "ib disabled");
  require(env.at("NCCL_P2P_DISABLE") == "1", "p2p disabled");
  require(env.at("NCCL_NVLS_ENABLE") == "0", "nvls disabled");
  require(env.at("NCCL_TOPO_FILE") == "/tmp/topo.xml", "topo file");
}

void test_pipeline_layer_ranges() {
  cverl::distributed::Topology topology(make_spec());
  auto r0 = topology.pipeline_layer_range(24, 0);
  auto r1 = topology.pipeline_layer_range(24, 1);
  require(r0.begin == 0 && r0.end == 12, "pipeline range 0");
  require(r1.begin == 12 && r1.end == 24, "pipeline range 1");

  auto uneven0 = topology.pipeline_layer_range(25, 0);
  auto uneven1 = topology.pipeline_layer_range(25, 1);
  require(uneven0.begin == 0 && uneven0.end == 13, "pipeline uneven range 0");
  require(uneven1.begin == 13 && uneven1.end == 25, "pipeline uneven range 1");

  auto local = topology.local_pipeline_layer_range(25);
  require(local.begin == 13 && local.end == 25, "local pipeline range");

  auto empty = topology.pipeline_layer_range(1, 1);
  require(empty.begin == 1 && empty.end == 1, "empty pipeline range");
}

void test_invalid_configs() {
  auto spec = make_spec();
  spec.world_size = 15;
  bool failed = false;
  try {
    cverl::distributed::Topology topology(spec);
  } catch (const std::invalid_argument&) {
    failed = true;
  }
  require(failed, "invalid world size rejected");

  spec = make_spec();
  spec.parallel.tensor_parallel = 16;
  spec.parallel.data_parallel = 1;
  spec.parallel.pipeline_parallel = 1;
  spec.world_size = 16;
  failed = false;
  try {
    cverl::distributed::Topology topology(spec);
  } catch (const std::invalid_argument&) {
    failed = true;
  }
  require(failed, "oversized tensor parallel rejected");

  spec = make_spec();
  spec.parallel.micro_batches = 1;
  failed = false;
  try {
    cverl::distributed::Topology topology(spec);
  } catch (const std::invalid_argument&) {
    failed = true;
  }
  require(failed, "too few micro batches rejected");

  cverl::distributed::Topology topology(make_spec());
  require_throws([&]() { (void)topology.global_rank(2, 0, 0); }, "global rank out-of-range rejected");
  require_throws([&]() { (void)topology.rank_info(16); }, "rank_info out-of-range rejected");
  require_throws([&]() { (void)topology.pipeline_layer_range(1, 2); }, "pipeline rank out-of-range rejected");
  require_throws([&]() { (void)topology.pipeline_layer_range(-1, 0); }, "negative layer count rejected");
}

void test_single_process_collectives() {
  cverl::distributed::SingleProcessCollectives collectives;
  auto x = torch::tensor({1.0f, 2.0f, 3.0f});
  auto y = collectives.all_reduce(x, cverl::distributed::ReduceOp::Sum, {0});
  require(torch::allclose(x, y), "single process all_reduce");
  auto z = collectives.all_gather(x, {0}, 0);
  require(torch::allclose(x, z), "single process all_gather");
  auto s = collectives.reduce_scatter(x, cverl::distributed::ReduceOp::Max, {0}, 0);
  require(torch::allclose(x, s), "single process reduce_scatter");
}

void test_cluster_spec_from_env() {
  setenv("WORLD_SIZE", "4", 1);
  setenv("RANK", "3", 1);
  setenv("LOCAL_WORLD_SIZE", "2", 1);
  setenv("LOCAL_RANK", "1", 1);
  setenv("NODE_RANK", "9", 1);
  setenv("CVERL_GPUS_PER_NODE", "4", 1);

  cverl::distributed::ParallelDims dims;
  dims.data_parallel = 2;
  dims.tensor_parallel = 2;
  dims.pipeline_parallel = 1;
  dims.micro_batches = 1;
  auto spec = cverl::distributed::cluster_spec_from_env(dims);
  require(spec.world_size == 4, "env world size");
  require(spec.rank == 3, "env rank");
  require(spec.local_world_size == 2, "env local world size");
  require(spec.local_rank == 1, "env local rank");
  require(spec.node_rank == 9, "env node rank");
  require(spec.gpus_per_node == 4, "env gpus per node");
  cverl::distributed::Topology topology(spec);
  auto info = topology.local_rank_info();
  require_vec_eq(info.tensor_group, {2, 3}, "env tensor group");
  require_vec_eq(info.data_group, {1, 3}, "env data group");
}

void test_dtype_names() {
  require(cverl::distributed::dtype_policy_name(cverl::distributed::DTypePolicy::Float32) == "float32", "float32 name");
  require(cverl::distributed::dtype_policy_name(cverl::distributed::DTypePolicy::Float16) == "float16", "float16 name");
  require(cverl::distributed::dtype_policy_name(cverl::distributed::DTypePolicy::BFloat16) == "bfloat16", "bfloat16 name");
}

}  // namespace

int main() {
  try {
    test_rank_mapping();
    test_nccl_env();
    test_pipeline_layer_ranges();
    test_invalid_configs();
    test_single_process_collectives();
    test_cluster_spec_from_env();
    test_dtype_names();
  } catch (const std::exception& e) {
    std::cerr << "test_topology failed: " << e.what() << "\n";
    return 1;
  }
  std::cout << "distributed topology tests passed\n";
  return 0;
}
