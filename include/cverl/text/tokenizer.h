#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cverl::text {

// Common encode options across all Tokenizer implementations. Backends that
// do not support a given knob must document the fallback behavior in their
// header (e.g. real BPE tokenizers ignore add_bos/add_eos when the model's
// post_processor template already inserts them).
struct EncodeOptions {
  // Prepend the BOS token (when one exists) to the encoded sequence.
  bool add_bos = false;
  // Append the EOS token (when one exists) to the encoded sequence.
  bool add_eos = false;
  // Truncate the result to this many tokens. -1 means no truncation. EOS is
  // emitted only when there is room left after truncation.
  int32_t max_tokens = -1;
};

// Abstract tokenizer interface. The trainer side, the rollout-batch builder,
// and any future component that turns text into model token ids should depend
// on this rather than a specific implementation. Concrete backends:
//
//   - ByteTokenizer       (CPU smoke / CI; byte-level, vocab=260)
//   - HfBpeTokenizer      (pure C++ HF byte-level BPE; loads tokenizer.json)
//   - (planned) HfTokenizersCpp (mlc-ai tokenizers-cpp wrapping HF Rust)
//
// Implementations must be safe to share across threads for read-only encode/
// decode after construction. They are not required to be const-correct on
// every member; a `const Tokenizer&` parameter is the canonical type.
class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  virtual std::vector<int32_t> encode(const std::string& text) const = 0;
  virtual std::vector<int32_t> encode(const std::string& text, const EncodeOptions& options) const = 0;

  // skip_special drops BOS/EOS/PAD/special added tokens from the output. When
  // false the implementation is allowed to render them as <bos>/<eos>/etc.
  // for debugging visibility.
  virtual std::string decode(const std::vector<int32_t>& ids, bool skip_special = true) const = 0;

  virtual int32_t vocab_size() const = 0;
  virtual int32_t pad_id() const = 0;
  virtual int32_t bos_id() const = 0;
  virtual int32_t eos_id() const = 0;
  // -1 when the backend has no notion of a generic unknown token id.
  virtual int32_t unk_id() const { return -1; }

  // Short identifier ("byte", "hf-bpe", ...).
  virtual std::string name() const = 0;
};

}  // namespace cverl::text
