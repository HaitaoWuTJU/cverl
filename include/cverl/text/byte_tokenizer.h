#pragma once

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
// On real GPU runs the GRPO trainer takes a `Tokenizer` interface (not yet
// implemented) and the HF tokenizer is plugged in there. The byte tokenizer
// is intended for tiny CPU smokes and unit tests, not production use.
class ByteTokenizer {
 public:
  static constexpr int32_t kPadId = 0;
  static constexpr int32_t kBosId = 1;
  static constexpr int32_t kEosId = 2;
  static constexpr int32_t kUnkId = 3;
  static constexpr int32_t kByteOffset = 4;
  static constexpr int32_t kVocabSize = 260;  // 4 specials + 256 bytes

  struct EncodeOptions {
    bool add_bos = false;
    bool add_eos = false;
    int32_t max_tokens = -1;  // -1 = unlimited
  };

  ByteTokenizer() = default;

  std::vector<int32_t> encode(const std::string& text) const;
  std::vector<int32_t> encode(const std::string& text, const EncodeOptions& options) const;
  std::string decode(const std::vector<int32_t>& ids, bool skip_special = true) const;

  static int32_t vocab_size() { return kVocabSize; }
  static int32_t pad_id() { return kPadId; }
  static int32_t bos_id() { return kBosId; }
  static int32_t eos_id() { return kEosId; }

  // True for ids that decode to literal bytes (kByteOffset..kByteOffset+255).
  static bool is_byte_id(int32_t id) { return id >= kByteOffset && id < kByteOffset + 256; }
};

}  // namespace cverl::text
