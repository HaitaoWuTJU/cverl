#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "cverl/distributed/pipeline.h"
#include "cverl/distributed/topology.h"

namespace {

int64_t arg_i64(int argc, char** argv, const std::string& flag, int64_t fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == flag) {
      return std::stoll(argv[i + 1]);
    }
  }
  return fallback;
}

bool has_flag(int argc, char** argv, const std::string& flag) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == flag) {
      return true;
    }
  }
  return false;
}

void print_vec(const char* name, const std::vector<int64_t>& values) {
  std::cout << name << "=";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << values[i];
  }
  std::cout << "\n";
}

void usage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " --dp D --tp T --pp P --cp C [--micro-batches M] [--rank R] [--world-size W]"
               " [--num-layers N] [--all-ranks]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (has_flag(argc, argv, "--help")) {
      usage(argv[0]);
      return 0;
    }

    cverl::distributed::ParallelDims dims;
    dims.data_parallel = arg_i64(argc, argv, "--dp", 1);
    dims.tensor_parallel = arg_i64(argc, argv, "--tp", 1);
    dims.pipeline_parallel = arg_i64(argc, argv, "--pp", 1);
    dims.context_parallel = arg_i64(argc, argv, "--cp", 1);
    dims.micro_batches = arg_i64(argc, argv, "--micro-batches", dims.pipeline_parallel);

    cverl::distributed::ClusterSpec spec = cverl::distributed::cluster_spec_from_env(dims);
    spec.world_size = arg_i64(argc, argv, "--world-size", spec.world_size);
    spec.rank = arg_i64(argc, argv, "--rank", spec.rank);
    spec.local_world_size = arg_i64(argc, argv, "--local-world-size", spec.local_world_size);
    spec.local_rank = arg_i64(argc, argv, "--local-rank", spec.local_rank);
    spec.gpus_per_node = arg_i64(argc, argv, "--gpus-per-node", spec.gpus_per_node);
    int64_t num_layers = arg_i64(argc, argv, "--num-layers", 0);

    cverl::distributed::Topology topology(spec);
    std::cout << "world_size=" << topology.world_size() << "\n";
    std::cout << "parallel=dp" << dims.data_parallel << "_pp" << dims.pipeline_parallel << "_cp"
              << dims.context_parallel << "_tp" << dims.tensor_parallel << "\n";
    std::cout << "micro_batches=" << dims.micro_batches << "\n";

    const bool all_ranks = has_flag(argc, argv, "--all-ranks");
    int64_t begin = all_ranks ? 0 : topology.rank();
    int64_t end = all_ranks ? topology.world_size() : topology.rank() + 1;
    for (int64_t rank = begin; rank < end; ++rank) {
      auto info = topology.rank_info(rank);
      auto peers = cverl::distributed::pipeline_peers(topology, info);
      std::cout << "rank=" << rank << " dp=" << info.data_rank << " pp=" << info.pipeline_rank
                << " cp=" << info.context_rank << " tp=" << info.tensor_rank << "\n";
      print_vec("  data_group", info.data_group);
      print_vec("  tensor_group", info.tensor_group);
      print_vec("  pipeline_group", info.pipeline_group);
      print_vec("  context_group", info.context_group);
      print_vec("  model_group", info.model_group);
      std::cout << "  pp_prev=" << peers.previous_rank << " pp_next=" << peers.next_rank
                << " pp_warmup_micro_batches=" << cverl::distributed::pipeline_warmup_micro_batches(topology, info)
                << "\n";
      if (num_layers > 0) {
        auto range = topology.pipeline_layer_range(num_layers, info.pipeline_rank);
        std::cout << "  layer_range=" << range.begin << ":" << range.end << "\n";
      }
    }

    auto env = topology.nccl_env();
    for (const auto& kv : env) {
      std::cout << "env." << kv.first << "=" << kv.second << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "cverl_parallel_plan failed: " << e.what() << "\n";
    usage(argv[0]);
    return 1;
  }
}
