#pragma once

#include <cstdint>

#include <torch/torch.h>

#include "cverl/distributed/collectives.h"
#include "cverl/distributed/topology.h"

namespace cverl::distributed {

struct PipelinePeers {
  int64_t previous_rank = -1;
  int64_t next_rank = -1;
  bool is_first_stage = true;
  bool is_last_stage = true;
};

PipelinePeers pipeline_peers(const Topology& topology, const ParallelRankInfo& info);

torch::Tensor pipeline_recv_forward(const torch::Tensor& like,
                                    Collectives& collectives,
                                    const PipelinePeers& peers);
void pipeline_send_forward(const torch::Tensor& activation,
                           Collectives& collectives,
                           const PipelinePeers& peers);
torch::Tensor pipeline_recv_backward(const torch::Tensor& like,
                                     Collectives& collectives,
                                     const PipelinePeers& peers);
void pipeline_send_backward(const torch::Tensor& grad,
                            Collectives& collectives,
                            const PipelinePeers& peers);

int64_t pipeline_warmup_micro_batches(const Topology& topology, const ParallelRankInfo& info);

}  // namespace cverl::distributed
