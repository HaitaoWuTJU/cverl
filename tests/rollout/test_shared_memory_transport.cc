// Tests SharedMemoryRolloutTransport via parent-client / forked-child-server.
// The child runs an "echo" server that copies prompts back and adds a few
// synthetic token ids, so we can assert that the wire format round-trips
// correctly across processes.
#include "cverl/rollout/shared_memory_transport.h"
#include "cverl/rollout/transport.h"

#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
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
    cverl::rollout::SharedMemoryTransportOptions opts;
    opts.name = "cverl_shm_xport_test_" + std::to_string(getpid());
    opts.payload_capacity = 256 * 1024;
    opts.timeout_ms = 5000;

    auto client = cverl::rollout::SharedMemoryRolloutTransport::create_client(opts);

    pid_t pid = fork();
    if (pid < 0) {
      throw std::runtime_error("fork failed");
    }
    if (pid == 0) {
      try {
        auto server = cverl::rollout::SharedMemoryRolloutTransport::attach_server(opts);
        for (int rt = 0; rt < 2; ++rt) {
          bool ok = server->serve_one([&](const cverl::rollout::RolloutRequest& req) {
            cverl::rollout::RolloutResponse resp;
            resp.request_id = req.request_id;
            for (uint32_t p = 0; p < req.prompts.size(); ++p) {
              for (uint32_t s = 0; s < req.n; ++s) {
                cverl::rollout::RolloutSequence seq;
                seq.prompt_index = p;
                seq.sample_index = s;
                seq.text = "echo:" + req.prompts[p] + ":" + std::to_string(s);
                seq.token_ids = {static_cast<int32_t>(p), static_cast<int32_t>(s),
                                 static_cast<int32_t>(req.seed)};
                seq.logprobs = {-0.1f, -0.2f, -0.3f};
                seq.finish_reason = "stop";
                resp.sequences.push_back(std::move(seq));
              }
            }
            resp.metrics["server_pid"] = std::to_string(getpid());
            return resp;
          });
          if (!ok) {
            std::_Exit(3);
          }
        }
        std::_Exit(0);
      } catch (const std::exception& e) {
        std::cerr << "child failed: " << e.what() << "\n";
        std::_Exit(2);
      }
    }

    cverl::rollout::RolloutRequest req;
    req.prompts = {"hello", "world"};
    req.n = 2;
    req.max_tokens = 16;
    req.seed = 42;
    req.return_token_ids = true;
    req.return_logprobs = true;
    req.stop = {"\n\n", "<eos>"};
    req.extra_params["repetition_penalty"] = "1.05";

    auto resp = client->generate(req);
    require(resp.sequences.size() == 4, "first round trip count");
    require(resp.sequences[0].text == "echo:hello:0", "first seq text");
    require(resp.sequences[3].text == "echo:world:1", "last seq text");
    require(resp.sequences[0].prompt_index == 0, "prompt_index 0");
    require(resp.sequences[2].prompt_index == 1, "prompt_index 1");
    require(resp.sequences[0].token_ids.size() == 3, "token ids round trip");
    require(resp.sequences[0].token_ids[2] == 42, "seed propagated");
    require(resp.sequences[0].logprobs.size() == 3, "logprobs round trip");
    require(resp.metrics.count("server_pid") == 1, "metrics round trip");

    // Second round trip uses a different request_id implicitly.
    cverl::rollout::RolloutRequest req2;
    req2.prompts = {"again"};
    req2.n = 1;
    req2.seed = 7;
    auto resp2 = client->generate(req2);
    require(resp2.sequences.size() == 1, "second round trip count");
    require(resp2.sequences[0].text == "echo:again:0", "second seq text");

    int status = 0;
    waitpid(pid, &status, 0);
    require(WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "child exited cleanly: status=" + std::to_string(status));

    std::cout << "shared memory rollout transport tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_shared_memory_transport failed: " << e.what() << "\n";
    return 1;
  }
}
