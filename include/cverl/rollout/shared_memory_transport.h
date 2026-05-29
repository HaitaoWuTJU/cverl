#pragma once

#include "cverl/rollout/transport.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace cverl::rollout {

// Options for SharedMemoryRolloutTransport. Both client and server must
// agree on the same `name` and buffer sizes.
struct SharedMemoryTransportOptions {
  // Common base name. Internally the transport derives:
  //   /<name>           -> shm region
  //   /<name>.req       -> POSIX semaphore signaling "request ready"
  //   /<name>.resp      -> POSIX semaphore signaling "response ready"
  std::string name;

  // Maximum bytes the request and response payloads may use, each. The
  // shared region size is sizeof(header) + 2 * payload_capacity. 8 MiB by
  // default is enough for ~50K of mid-sized GSM8K responses; bump for
  // larger batches.
  size_t payload_capacity = 8 * 1024 * 1024;

  // Per round-trip timeout in milliseconds. Generate() throws on timeout.
  long timeout_ms = 60000;
};

// Round-trip rollout transport on top of POSIX shared memory + named
// semaphores. The producing side (LLM rollout sidecar) creates a server with
// `attach_server`, the consumer side (trainer) creates a client with
// `create_client`. Currently single-slot blocking: the next call to
// `generate` does not return until the server has processed the previous
// one. Multi-slot ring buffer can plug in behind the same RolloutTransport
// interface without changing trainer code.
class SharedMemoryRolloutTransport : public RolloutTransport {
 public:
  // Allocate the shm region + semaphores. Caller is responsible for keeping
  // the returned object alive until the server attaches and finishes.
  static std::unique_ptr<SharedMemoryRolloutTransport> create_client(
      const SharedMemoryTransportOptions& options);

  // Attach to an existing region created by `create_client`.
  static std::unique_ptr<SharedMemoryRolloutTransport> attach_server(
      const SharedMemoryTransportOptions& options);

  ~SharedMemoryRolloutTransport() override;

  SharedMemoryRolloutTransport(const SharedMemoryRolloutTransport&) = delete;
  SharedMemoryRolloutTransport& operator=(const SharedMemoryRolloutTransport&) = delete;

  // Client API.
  RolloutResponse generate(const RolloutRequest& request) override;
  std::string name() const override;

  // Server API: block until the client posts a request, run `handler` on
  // the deserialized request, and post the response back. Returns false on
  // timeout. `handler` runs synchronously in the calling thread.
  using Handler = std::function<RolloutResponse(const RolloutRequest&)>;
  bool serve_one(const Handler& handler);

  // Server API: loop calling serve_one until *running becomes false. Useful
  // for sidecar processes.
  void serve_loop(const Handler& handler, std::atomic<bool>& running);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  bool is_client_ = false;

  SharedMemoryRolloutTransport();
};

}  // namespace cverl::rollout
