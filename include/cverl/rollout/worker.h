#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

#include "cverl/distributed/collectives.h"
#include "cverl/distributed/weight_sync.h"

namespace cverl::rollout {

struct TokenBatch {
  torch::Tensor token_ids;       // int64 [B, T], CUDA preferred
  torch::Tensor attention_mask;  // int64/bool/float [B, T], optional
  std::vector<int64_t> lengths;  // optional CPU lengths
};

struct GenerationConfig {
  int64_t max_new_tokens = 256;
  double temperature = 1.0;
  double top_p = 1.0;
  int64_t top_k = -1;
  int64_t eos_token_id = -1;
  // 0 disables per-token host synchronization for early-exit checks. Set to 1
  // for lowest-latency early exit, or a larger interval to amortize the sync.
  int64_t eos_check_interval = 0;
  uint64_t seed = 0;
};

struct GenerationOutput {
  torch::Tensor token_ids;  // int64 [B, max_new_tokens], CUDA preferred
  torch::Tensor logprobs;   // float [B, max_new_tokens], optional
  torch::Tensor lengths;    // int64 [B], optional
};

// Core rollout boundary. Efficient integrations should implement this inside a
// vLLM worker, Megatron worker, or native C++/CUDA rollout worker so generation
// and actor parameter tensors stay on GPU. HTTP adapters should not implement
// this interface.
class RolloutWorker {
 public:
  virtual ~RolloutWorker() = default;

  virtual GenerationOutput generate(const TokenBatch& prompts,
                                    const GenerationConfig& config) = 0;

  // Return rollout-side actor weight tensors in the same order/names as the
  // trainer-side policy. These tensors are synchronized directly via NCCL or
  // CUDA IPC; no checkpoint serialization is involved.
  virtual std::vector<cverl::distributed::ParameterView> actor_parameters() = 0;
};

void synchronize_rollout_actor_weights(
    RolloutWorker& worker,
    cverl::distributed::Collectives& collectives,
    int64_t trainer_root,
    const std::vector<int64_t>& group);

}  // namespace cverl::rollout
