#include "cverl/rollout/transport.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool cond, const std::string& msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

}  // namespace

int main() {
  try {
    cverl::rollout::LoopbackRolloutTransport::Options opts;
    opts.completion_suffix = " #### 7";
    cverl::rollout::LoopbackRolloutTransport transport(opts);

    cverl::rollout::RolloutRequest req;
    req.prompts = {"What is 1+1?", "What is 2+2?"};
    req.n = 3;
    req.max_tokens = 16;
    req.return_token_ids = true;
    req.seed = 42;

    auto resp = transport.generate(req);
    require(resp.sequences.size() == 6, "loopback returns prompts*n sequences");
    require(transport.name() == "loopback", "transport name");

    // Sequences come back in (prompt_index, sample_index) order.
    for (size_t i = 0; i < resp.sequences.size(); ++i) {
      const auto& seq = resp.sequences[i];
      uint32_t expected_prompt = static_cast<uint32_t>(i / req.n);
      uint32_t expected_sample = static_cast<uint32_t>(i % req.n);
      require(seq.prompt_index == expected_prompt, "prompt_index ordering");
      require(seq.sample_index == expected_sample, "sample_index ordering");
      require(seq.text.find("####") != std::string::npos, "completion suffix appended");
      require(!seq.token_ids.empty(), "token_ids requested");
      require(seq.finish_reason == "stop", "finish_reason set");
    }

    // Different seeds for the same prompt should produce different token ids.
    require(resp.sequences[0].token_ids != resp.sequences[1].token_ids,
            "different sample_index yields different ids");

    std::cout << "rollout transport (loopback) tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_transport failed: " << e.what() << "\n";
    return 1;
  }
}
