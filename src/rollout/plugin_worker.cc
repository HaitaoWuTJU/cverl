#include "cverl/rollout/plugin_worker.h"

#include <dlfcn.h>

#include <stdexcept>
#include <utility>

namespace cverl::rollout {
namespace {

void* checked_symbol(void* handle, const char* name) {
  dlerror();
  void* sym = dlsym(handle, name);
  const char* err = dlerror();
  if (err != nullptr || sym == nullptr) {
    throw std::runtime_error(std::string("rollout plugin missing symbol ") + name +
                             (err != nullptr ? (": " + std::string(err)) : ""));
  }
  return sym;
}

}  // namespace

DynamicRolloutWorker::DynamicRolloutWorker(std::string shared_library_path,
                                           std::string config_json) {
  handle_ = dlopen(shared_library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle_ == nullptr) {
    throw std::runtime_error("failed to load rollout plugin " + shared_library_path +
                             ": " + dlerror());
  }

  auto create = reinterpret_cast<CreateRolloutWorkerFn>(
      checked_symbol(handle_, "cverl_create_rollout_worker"));
  destroy_ = reinterpret_cast<DestroyRolloutWorkerFn>(
      checked_symbol(handle_, "cverl_destroy_rollout_worker"));
  worker_ = create(config_json.c_str());
  if (worker_ == nullptr) {
    throw std::runtime_error("rollout plugin returned null worker: " + shared_library_path);
  }
}

DynamicRolloutWorker::~DynamicRolloutWorker() {
  if (worker_ != nullptr && destroy_ != nullptr) {
    destroy_(worker_);
  }
  if (handle_ != nullptr) {
    dlclose(handle_);
  }
}

GenerationOutput DynamicRolloutWorker::generate(const TokenBatch& prompts,
                                                const GenerationConfig& config) {
  return worker_->generate(prompts, config);
}

std::vector<cverl::distributed::ParameterView> DynamicRolloutWorker::actor_parameters() {
  return worker_->actor_parameters();
}

std::unique_ptr<RolloutWorker> load_rollout_worker_plugin(
    const std::string& shared_library_path,
    const std::string& config_json) {
  return std::make_unique<DynamicRolloutWorker>(shared_library_path, config_json);
}

}  // namespace cverl::rollout
