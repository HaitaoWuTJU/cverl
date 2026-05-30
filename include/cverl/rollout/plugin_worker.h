#pragma once

#include <memory>
#include <string>

#include "cverl/rollout/worker.h"

namespace cverl::rollout {

// Factory symbol exported by high-performance rollout plugins.
//
// A vLLM/Megatron plugin should create a RolloutWorker that directly owns or
// references engine-side GPU token/weight tensors. The returned worker must not
// route generation through HTTP, JSON, or checkpoint reloads.
//
// extern "C" cverl::rollout::RolloutWorker* cverl_create_rollout_worker(
//     const char* config_json);
// extern "C" void cverl_destroy_rollout_worker(cverl::rollout::RolloutWorker*);
using CreateRolloutWorkerFn = RolloutWorker* (*)(const char* config_json);
using DestroyRolloutWorkerFn = void (*)(RolloutWorker* worker);

class DynamicRolloutWorker final : public RolloutWorker {
 public:
  DynamicRolloutWorker(std::string shared_library_path, std::string config_json);
  ~DynamicRolloutWorker() override;

  DynamicRolloutWorker(const DynamicRolloutWorker&) = delete;
  DynamicRolloutWorker& operator=(const DynamicRolloutWorker&) = delete;
  DynamicRolloutWorker(DynamicRolloutWorker&&) = delete;
  DynamicRolloutWorker& operator=(DynamicRolloutWorker&&) = delete;

  GenerationOutput generate(const TokenBatch& prompts,
                            const GenerationConfig& config) override;

  std::vector<cverl::distributed::ParameterView> actor_parameters() override;

 private:
  void* handle_ = nullptr;
  DestroyRolloutWorkerFn destroy_ = nullptr;
  RolloutWorker* worker_ = nullptr;
};

std::unique_ptr<RolloutWorker> load_rollout_worker_plugin(
    const std::string& shared_library_path,
    const std::string& config_json);

}  // namespace cverl::rollout
