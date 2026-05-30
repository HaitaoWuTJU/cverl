#include "cverl/distributed/pipeline.h"

#include <algorithm>
#include <stdexcept>

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

std::vector<PipelineScheduleAction> pipeline_1f1b_schedule(const Topology& topology,
                                                           const ParallelRankInfo& info) {
  const auto& p = topology.spec().parallel;
  if (p.micro_batches < p.pipeline_parallel && p.pipeline_parallel > 1) {
    throw std::invalid_argument("1F1B requires micro_batches >= pipeline_parallel");
  }

  const auto peers = pipeline_peers(topology, info);
  const int64_t warmup = pipeline_warmup_micro_batches(topology, info);
  const int64_t steady = p.micro_batches - warmup;
  std::vector<PipelineScheduleAction> actions;
  actions.reserve(static_cast<size_t>(warmup + steady * 2 + warmup));

  for (int64_t mb = 0; mb < warmup; ++mb) {
    actions.push_back(PipelineScheduleAction{
        PipelineSchedulePhase::Warmup,
        PipelineScheduleOp::Forward,
        mb,
        !peers.is_first_stage,
        !peers.is_last_stage,
        false,
        false,
    });
  }

  for (int64_t i = 0; i < steady; ++i) {
    actions.push_back(PipelineScheduleAction{
        PipelineSchedulePhase::Steady,
        PipelineScheduleOp::Forward,
        warmup + i,
        !peers.is_first_stage,
        !peers.is_last_stage,
        false,
        false,
    });
    actions.push_back(PipelineScheduleAction{
        PipelineSchedulePhase::Steady,
        PipelineScheduleOp::Backward,
        i,
        false,
        false,
        !peers.is_last_stage,
        !peers.is_first_stage,
    });
  }

  for (int64_t mb = steady; mb < p.micro_batches; ++mb) {
    actions.push_back(PipelineScheduleAction{
        PipelineSchedulePhase::Cooldown,
        PipelineScheduleOp::Backward,
        mb,
        false,
        false,
        !peers.is_last_stage,
        !peers.is_first_stage,
    });
  }
  return actions;
}

int64_t pipeline_schedule_max_live_activations(const std::vector<PipelineScheduleAction>& actions) {
  int64_t live = 0;
  int64_t max_live = 0;
  for (const auto& action : actions) {
    if (action.op == PipelineScheduleOp::Forward) {
      ++live;
      max_live = std::max(max_live, live);
    } else {
      if (live <= 0) {
        throw std::invalid_argument("invalid 1F1B schedule: backward before matching forward");
      }
      --live;
    }
  }
  if (live != 0) {
    throw std::invalid_argument("invalid 1F1B schedule: unmatched forward activations");
  }
  return max_live;
}

const char* pipeline_schedule_phase_name(PipelineSchedulePhase phase) {
  switch (phase) {
    case PipelineSchedulePhase::Warmup:
      return "warmup";
    case PipelineSchedulePhase::Steady:
      return "steady";
    case PipelineSchedulePhase::Cooldown:
      return "cooldown";
  }
  return "unknown";
}

const char* pipeline_schedule_op_name(PipelineScheduleOp op) {
  switch (op) {
    case PipelineScheduleOp::Forward:
      return "forward";
    case PipelineScheduleOp::Backward:
      return "backward";
  }
  return "unknown";
}

}  // namespace cverl::distributed
