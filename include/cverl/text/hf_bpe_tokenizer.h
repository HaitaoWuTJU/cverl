#pragma once

#include "cverl/text/tokenizer.h"

#include <memory>
#include <string>
#include <vector>

namespace cverl::text {

struct HfBpeTokenizerOptions {
  // Path to a HuggingFace tokenizer.json file.
  std::string tokenizer_json_path;

  // Optional explicit pad token id. When < 0, the tokenizer first looks for
  // an added_tokens entry whose content matches `pad_token_string` (default
  // "<|pad|>"), then falls back to the EOS id. This mirrors how HF's
  // PreTrainedTokenizer auto-discovers pad ids when they are not declared.
  int32_t pad_id = -1;
  std::string pad_token_string = "<|pad|>";
};

// Pure C++ HuggingFace byte-level BPE tokenizer. Loads a tokenizer.json file
// (model.type == "BPE", byte-level pre_tokenizer, byte-level decoder) and
// runs the canonical BPE algorithm. Targets Qwen / GPT-2 / Llama-3 family
// tokenizers. SentencePiece (Llama-2 / Mistral) and other model types are
// rejected with a clear error.
//
// Compatibility notes:
//   - Vocab token strings use the GPT-2 byte_to_unicode mapping (256 bytes
//     mapped to 256 unicode codepoints). This implementation reproduces the
//     same mapping bit-for-bit.
//   - Pre-tokenization uses the regex declared in the tokenizer.json
//     pre_tokenizer (Qwen3 / GPT-2 use a tiktoken-style regex).
//   - added_tokens / special tokens are matched as exact strings before BPE,
//     so input segments containing a special are split around it and the
//     special is emitted with its literal id.
//   - Decoding reverses the byte mapping; non-skip mode renders specials
//     verbatim.
//
// Performance notes:
//   - Encoding cost is O(N * M) per pre-token where N is the byte length and
//     M is the number of remaining merge candidates. Replace the inner loop
//     with a priority queue / linked-list cache later if profiling shows
//     this is hot.
//   - All lookups are unordered_maps with reserve() at construction time;
//     no allocations on the hot path beyond the merge buffers.
//
// Thread-safety: encode()/decode() are const and reentrant after
// construction.
class HfBpeTokenizer final : public Tokenizer {
 public:
  explicit HfBpeTokenizer(const HfBpeTokenizerOptions& options);
  ~HfBpeTokenizer() override;

  HfBpeTokenizer(const HfBpeTokenizer&) = delete;
  HfBpeTokenizer& operator=(const HfBpeTokenizer&) = delete;

  // Convenience: load tokenizer.json from <model_dir>/tokenizer.json.
  static std::unique_ptr<HfBpeTokenizer> from_model_dir(const std::string& model_dir);

  std::vector<int32_t> encode(const std::string& text) const override;
  std::vector<int32_t> encode(const std::string& text, const EncodeOptions& options) const override;
  std::string decode(const std::vector<int32_t>& ids, bool skip_special = true) const override;

  int32_t vocab_size() const override;
  int32_t pad_id() const override;
  int32_t bos_id() const override;
  int32_t eos_id() const override;
  int32_t unk_id() const override;
  std::string name() const override { return "hf-bpe"; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cverl::text
