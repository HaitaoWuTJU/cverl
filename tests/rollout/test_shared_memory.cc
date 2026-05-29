#include "cverl/rollout/shared_memory.h"

#include <sys/wait.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool cond, const char* msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

struct Payload {
  cverl::rollout::SharedRolloutHeader header;
  int32_t tokens[8];
};

}  // namespace

int main() {
  try {
    std::string name = cverl::rollout::SharedMemoryRegion::make_name("cverl_rollout_test");
    cverl::rollout::SharedMemoryRegion parent(name, sizeof(Payload), cverl::rollout::SharedMemoryOpenMode::Create);
    std::memset(parent.data(), 0, parent.size());
    auto* p = parent.as<Payload>();
    p->header = cverl::rollout::SharedRolloutHeader{};
    p->header.request_id = 42;
    p->header.prompt_count = 2;
    p->header.max_prompt_tokens = 4;
    for (int i = 0; i < 8; ++i) {
      p->tokens[i] = i;
    }

    pid_t pid = fork();
    if (pid < 0) {
      throw std::runtime_error("fork failed");
    }
    if (pid == 0) {
      try {
        cverl::rollout::SharedMemoryRegion child(
            name, sizeof(Payload), cverl::rollout::SharedMemoryOpenMode::OpenExisting);
        auto* q = child.as<Payload>();
        if (q->header.magic != 0x4356524c || q->header.request_id != 42 || q->tokens[7] != 7) {
          return 3;
        }
        q->header.status = 7;
        q->tokens[0] = 99;
        return 0;
      } catch (...) {
        return 2;
      }
    }

    int status = 0;
    waitpid(pid, &status, 0);
    require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "child process failed");
    require(p->header.status == 7, "child status write not visible");
    require(p->tokens[0] == 99, "child token write not visible");
    parent.unlink();

    std::cout << "shared rollout memory tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_shared_memory failed: " << e.what() << "\n";
    return 1;
  }
}
