#include "cverl/distributed/pipeline.h"

#include <algorithm>

namespace cverl::distributed {

PipelinePeers pipeline_peers(const Topology& topology, const ParallelRankInfo& info) {
  const auto& p = topology.spec().parallel;
  PipelinePeers peers;
  peers.is_first_stage = info.pipeline_rank == 0;
  peers.is_last_stage = info.pipeline_rank + 1 == p.pipeline_parallel;
  if (!peers.is_first_stage) {
    peers.previous_rank =
        topology.global_rank(info.data_rank, info.pipeline_rank - 1, info.context_rank, info.tensor_rank);
  }
  if (!peers.is_last_stage) {
    peers.next_rank = topology.global_rank(info.data_rank, info.pipeline_rank + 1, info.context_rank, info.tensor_rank);
  }
  return peers;
}

torch::Tensor pipeline_recv_forward(const torch::Tensor& like,
                                    Collectives& collectives,
                                    const PipelinePeers& peers) {
  if (peers.is_first_stage) {
    return like;
  }
  return collectives.recv_like(like, peers.previous_rank);
}

void pipeline_send_forward(const torch::Tensor& activation,
                           Collectives& collectives,
                           const PipelinePeers& peers) {
  if (!peers.is_last_stage) {
    collectives.send(activation.contiguous(), peers.next_rank);
  }
}

torch::Tensor pipeline_recv_backward(const torch::Tensor& like,
                                     Collectives& collectives,
                                     const PipelinePeers& peers) {
  if (peers.is_last_stage) {
    return like;
  }
  return collectives.recv_like(like, peers.next_rank);
}

void pipeline_send_backward(const torch::Tensor& grad,
                            Collectives& collectives,
                            const PipelinePeers& peers) {
  if (!peers.is_first_stage) {
    collectives.send(grad.contiguous(), peers.previous_rank);
  }
}

int64_t pipeline_warmup_micro_batches(const Topology& topology, const ParallelRankInfo& info) {
  const auto& p = topology.spec().parallel;
  return std::min(p.pipeline_parallel - info.pipeline_rank - 1, p.micro_batches);
}

}  // namespace cverl::distributed
