#include "cverl/rollout/plugin_worker.h"

#include <iostream>
#include <stdexcept>

namespace {

void require(bool cond, const std::string& msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc >= 2, "usage: test_plugin_worker <plugin.so>");
    auto worker = cverl::rollout::load_rollout_worker_plugin(argv[1], "{\"backend\":\"test\"}");

    cverl::rollout::TokenBatch prompts;
    prompts.token_ids = torch::ones({2, 4}, torch::kLong);
    cverl::rollout::GenerationConfig config;
    config.max_new_tokens = 3;
    auto out = worker->generate(prompts, config);
    require(out.token_ids.dim() == 2 && out.token_ids.size(0) == 2 && out.token_ids.size(1) == 3,
            "plugin token shape");
    require(!worker->actor_parameters().empty(), "plugin actor parameters");

    std::cout << "plugin rollout worker tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_plugin_worker failed: " << e.what() << "\n";
    return 1;
  }
}
