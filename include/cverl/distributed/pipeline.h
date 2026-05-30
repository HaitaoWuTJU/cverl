#pragma once

#include <cstdint>
#include <vector>

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

enum class PipelineSchedulePhase {
  Warmup,
  Steady,
  Cooldown,
};

enum class PipelineScheduleOp {
  Forward,
  Backward,
};

struct PipelineScheduleAction {
  PipelineSchedulePhase phase = PipelineSchedulePhase::Warmup;
  PipelineScheduleOp op = PipelineScheduleOp::Forward;
  int64_t micro_batch = 0;
  bool recv_forward = false;
  bool send_forward = false;
  bool recv_backward = false;
  bool send_backward = false;
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
std::vector<PipelineScheduleAction> pipeline_1f1b_schedule(const Topology& topology,
                                                           const ParallelRankInfo& info);
int64_t pipeline_schedule_max_live_activations(const std::vector<PipelineScheduleAction>& actions);

const char* pipeline_schedule_phase_name(PipelineSchedulePhase phase);
const char* pipeline_schedule_op_name(PipelineScheduleOp op);

}  // namespace cverl::distributed
