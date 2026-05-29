#include "cverl/text/byte_tokenizer.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cverl::text {

std::vector<int32_t> ByteTokenizer::encode(const std::string& text) const {
  return encode(text, EncodeOptions{});
}

std::vector<int32_t> ByteTokenizer::encode(const std::string& text, const EncodeOptions& options) const {
  std::vector<int32_t> ids;
  size_t cap = text.size() + (options.add_bos ? 1 : 0) + (options.add_eos ? 1 : 0);
  ids.reserve(cap);
  if (options.add_bos) {
    ids.push_back(kBosId);
  }
  for (unsigned char c : text) {
    if (options.max_tokens > 0 && static_cast<int32_t>(ids.size()) >= options.max_tokens) {
      break;
    }
    ids.push_back(kByteOffset + static_cast<int32_t>(c));
  }
  if (options.add_eos && (options.max_tokens <= 0 || static_cast<int32_t>(ids.size()) < options.max_tokens)) {
    ids.push_back(kEosId);
  }
  return ids;
}

std::string ByteTokenizer::decode(const std::vector<int32_t>& ids, bool skip_special) const {
  std::string out;
  out.reserve(ids.size());
  for (int32_t id : ids) {
    if (id < 0 || id >= kVocabSize) {
      continue;
    }
    if (id < kByteOffset) {
      if (skip_special) {
        continue;
      }
      // Round-trip non-skip mode: emit a placeholder so callers can detect
      // specials without losing alignment.
      switch (id) {
        case kPadId:
          out.push_back('\0');
          break;
        case kBosId:
          out.append("<bos>");
          break;
        case kEosId:
          out.append("<eos>");
          break;
        case kUnkId:
          out.append("<unk>");
          break;
        default:
          break;
      }
      continue;
    }
    out.push_back(static_cast<char>(static_cast<unsigned char>(id - kByteOffset)));
  }
  return out;
}

}  // namespace cverl::text
