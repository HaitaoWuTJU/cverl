#include "cverl/rollout/transport.h"

#include <cstdint>
#include <string>
#include <utility>

namespace cverl::rollout {

namespace {

uint64_t fnv1a64(const std::string& text, uint64_t seed) {
  uint64_t hash = 1469598103934665603ULL ^ seed;
  for (unsigned char ch : text) {
    hash ^= ch;
    hash *= 1099511628211ULL;
  }
  return hash;
}

}  // namespace

LoopbackRolloutTransport::LoopbackRolloutTransport() = default;

LoopbackRolloutTransport::LoopbackRolloutTransport(Options options) : options_(std::move(options)) {}

RolloutResponse LoopbackRolloutTransport::generate(const RolloutRequest& request) {
  RolloutResponse response;
  response.request_id = request.request_id;
  response.sequences.reserve(static_cast<size_t>(request.prompts.size()) * request.n);
  for (size_t p = 0; p < request.prompts.size(); ++p) {
    for (uint32_t s = 0; s < request.n; ++s) {
      RolloutSequence seq;
      seq.prompt_index = static_cast<uint32_t>(p);
      seq.sample_index = s;
      seq.text = request.prompts[p];
      if (!options_.completion_suffix.empty()) {
        seq.text += options_.completion_suffix;
      }
      if (options_.synthesize_token_ids || request.return_token_ids) {
        uint64_t state = fnv1a64(seq.text, request.seed + s);
        size_t length = std::min<size_t>(request.max_tokens, seq.text.size() + 1);
        seq.token_ids.reserve(length);
        for (size_t i = 0; i < length; ++i) {
          state ^= state >> 12;
          state ^= state << 25;
          state ^= state >> 27;
          seq.token_ids.push_back(static_cast<int32_t>(state & 0x7FFFFFFFull));
        }
      }
      seq.finish_reason = "stop";
      response.sequences.push_back(std::move(seq));
    }
  }
  return response;
}

}  // namespace cverl::rollout
