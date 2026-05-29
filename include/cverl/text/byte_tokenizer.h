#pragma once

#include "cverl/text/tokenizer.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cverl::text {

// ByteTokenizer is a deterministic, dependency-free fallback tokenizer used
// for CPU smoke tests where a real HF tokenizer is not available. Vocabulary
// layout:
//   0   - pad token
//   1   - bos token
//   2   - eos token
//   3   - unk token  (reserved; never emitted by encode())
//   4..259 - one id per raw byte 0..255
//
// The byte tokenizer is intended for tiny CPU smokes and unit tests, not
// production use. For real model training, plug HfBpeTokenizer (or in future
// the tokenizers-cpp wrapper) behind the Tokenizer interface so token ids
// match those returned by the rollout server.
class ByteTokenizer final : public Tokenizer {
 public:
  static constexpr int32_t kPadId = 0;
  static constexpr int32_t kBosId = 1;
  static constexpr int32_t kEosId = 2;
  static constexpr int32_t kUnkId = 3;
  static constexpr int32_t kByteOffset = 4;
  static constexpr int32_t kVocabSize = 260;  // 4 specials + 256 bytes

  ByteTokenizer() = default;

  std::vector<int32_t> encode(const std::string& text) const override;
  std::vector<int32_t> encode(const std::string& text, const EncodeOptions& options) const override;
  std::string decode(const std::vector<int32_t>& ids, bool skip_special = true) const override;

  int32_t vocab_size() const override { return kVocabSize; }
  int32_t pad_id() const override { return kPadId; }
  int32_t bos_id() const override { return kBosId; }
  int32_t eos_id() const override { return kEosId; }
  int32_t unk_id() const override { return kUnkId; }
  std::string name() const override { return "byte"; }

  // True for ids that decode to literal bytes (kByteOffset..kByteOffset+255).
  static bool is_byte_id(int32_t id) { return id >= kByteOffset && id < kByteOffset + 256; }
};

}  // namespace cverl::text
