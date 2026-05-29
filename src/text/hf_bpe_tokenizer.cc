#include "cverl/text/hf_bpe_tokenizer.h"

#include <pcre2posix.h>
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cverl::text {

namespace {

using json = nlohmann::json;

// Hash for std::pair<std::string, std::string>.
struct StringPairHash {
  size_t operator()(const std::pair<std::string, std::string>& p) const noexcept {
    auto h1 = std::hash<std::string>{}(p.first);
    auto h2 = std::hash<std::string>{}(p.second);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

// HF GPT-2 byte_to_unicode mapping. Returns a [256]-sized array of unicode
// codepoints, one per byte 0..255. This is the canonical table used by
// GPT-2 / Qwen / Llama-3 byte-level BPE.
std::array<uint32_t, 256> build_byte_to_unicode_table() {
  std::array<uint32_t, 256> table{};
  std::vector<uint8_t> bs;
  bs.reserve(188);
  for (int c = 0x21; c <= 0x7E; ++c) bs.push_back(static_cast<uint8_t>(c));
  for (int c = 0xA1; c <= 0xAC; ++c) bs.push_back(static_cast<uint8_t>(c));
  for (int c = 0xAE; c <= 0xFF; ++c) bs.push_back(static_cast<uint8_t>(c));
  std::vector<uint32_t> cs(bs.begin(), bs.end());
  uint32_t n = 0;
  std::array<bool, 256> in_bs{};
  for (auto b : bs) in_bs[b] = true;
  for (int b = 0; b < 256; ++b) {
    if (!in_bs[b]) {
      bs.push_back(static_cast<uint8_t>(b));
      cs.push_back(256u + n);
      ++n;
    }
  }
  for (size_t i = 0; i < bs.size(); ++i) {
    table[bs[i]] = cs[i];
  }
  return table;
}

const std::array<uint32_t, 256>& byte_to_unicode() {
  static const std::array<uint32_t, 256> table = build_byte_to_unicode_table();
  return table;
}

// Encode one unicode codepoint as UTF-8 bytes appended to `out`.
void utf8_append(uint32_t cp, std::string& out) {
  if (cp <= 0x7F) {
    out.push_back(static_cast<char>(cp));
  } else if (cp <= 0x7FF) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp <= 0xFFFF) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// Decode UTF-8 bytes into a sequence of codepoints. Returns false on
// malformed input; partial result is still in `out`.
bool utf8_decode(const std::string& s, std::vector<uint32_t>& out) {
  size_t i = 0;
  while (i < s.size()) {
    uint8_t b0 = static_cast<uint8_t>(s[i]);
    uint32_t cp = 0;
    size_t advance = 0;
    if (b0 < 0x80) {
      cp = b0;
      advance = 1;
    } else if ((b0 & 0xE0) == 0xC0 && i + 1 < s.size()) {
      cp = (b0 & 0x1F) << 6;
      cp |= static_cast<uint8_t>(s[i + 1]) & 0x3F;
      advance = 2;
    } else if ((b0 & 0xF0) == 0xE0 && i + 2 < s.size()) {
      cp = (b0 & 0x0F) << 12;
      cp |= (static_cast<uint8_t>(s[i + 1]) & 0x3F) << 6;
      cp |= static_cast<uint8_t>(s[i + 2]) & 0x3F;
      advance = 3;
    } else if ((b0 & 0xF8) == 0xF0 && i + 3 < s.size()) {
      cp = (b0 & 0x07) << 18;
      cp |= (static_cast<uint8_t>(s[i + 1]) & 0x3F) << 12;
      cp |= (static_cast<uint8_t>(s[i + 2]) & 0x3F) << 6;
      cp |= static_cast<uint8_t>(s[i + 3]) & 0x3F;
      advance = 4;
    } else {
      return false;
    }
    out.push_back(cp);
    i += advance;
  }
  return true;
}

// Convert a UTF-8 byte string (raw input text) into the byte-level unicode
// representation used by HF BPE: each input byte becomes its byte_to_unicode
// codepoint, then encoded as UTF-8. The result is the form the BPE
// algorithm and the vocab map operate on.
std::string bytes_to_unicode_string(const std::string& raw) {
  const auto& tbl = byte_to_unicode();
  std::string out;
  out.reserve(raw.size() * 2);
  for (unsigned char b : raw) {
    utf8_append(tbl[b], out);
  }
  return out;
}

// PCRE2 wrapper. Owns the compiled pattern + match data; thin wrapper over
// pcre2_match for repeated splitting.
class Pcre2Regex {
 public:
  explicit Pcre2Regex(const std::string& pattern) {
    int errornumber = 0;
    PCRE2_SIZE erroroffset = 0;
    code_ = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.data()),
                          pattern.size(),
                          PCRE2_UTF | PCRE2_UCP,
                          &errornumber,
                          &erroroffset,
                          nullptr);
    if (code_ == nullptr) {
      PCRE2_UCHAR buf[256];
      pcre2_get_error_message(errornumber, buf, sizeof(buf));
      throw std::runtime_error(std::string("PCRE2 compile failed: ") +
                               reinterpret_cast<const char*>(buf) +
                               " in pattern: " + pattern);
    }
    match_data_ = pcre2_match_data_create_from_pattern(code_, nullptr);
  }
  ~Pcre2Regex() {
    if (match_data_) pcre2_match_data_free(match_data_);
    if (code_) pcre2_code_free(code_);
  }
  Pcre2Regex(const Pcre2Regex&) = delete;
  Pcre2Regex& operator=(const Pcre2Regex&) = delete;

  // Find every consecutive match in `text`, append the matched substring to
  // `out`. This is the "regex split into tokens" step HF's ByteLevel
  // pre_tokenizer performs.
  void find_all(const std::string& text, std::vector<std::string>& out) const {
    PCRE2_SIZE start = 0;
    while (start <= text.size()) {
      int rc = pcre2_match(code_,
                           reinterpret_cast<PCRE2_SPTR>(text.data()),
                           text.size(),
                           start,
                           0,
                           match_data_,
                           nullptr);
      if (rc < 0) break;
      PCRE2_SIZE* ovec = pcre2_get_ovector_pointer(match_data_);
      PCRE2_SIZE m_start = ovec[0];
      PCRE2_SIZE m_end = ovec[1];
      if (m_end == m_start) {
        // Zero-length match: advance one byte to avoid infinite loop.
        ++start;
        continue;
      }
      out.emplace_back(text.data() + m_start, m_end - m_start);
      start = m_end;
    }
  }

 private:
  pcre2_code* code_ = nullptr;
  pcre2_match_data* match_data_ = nullptr;
};

// Default regex used when tokenizer.json's pre_tokenizer does not declare
// a pattern. Matches the GPT-2 / Qwen3 tiktoken-style pattern.
constexpr const char* kQwenLikePattern =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
    "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|"
    "\\p{N}{1,3}|"
    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
    "\\s*[\\r\\n]+|"
    "\\s+(?!\\S)|"
    "\\s+";

}  // namespace

struct HfBpeTokenizer::Impl {
  // Forward (token text -> id) and reverse (id -> token text) vocab.
  std::unordered_map<std::string, int32_t> vocab;
  std::vector<std::string> id_to_token;
  // BPE merge ranks: lower = applied first.
  std::unordered_map<std::pair<std::string, std::string>, int32_t, StringPairHash> merge_ranks;

  // Pre-tokenizer regex.
  std::unique_ptr<Pcre2Regex> pre_tokenizer;

  // Special / added tokens. content -> id, sorted by descending content
  // length so the longest match wins when scanning the input.
  std::vector<std::pair<std::string, int32_t>> added_tokens_sorted;
  // Set of ids that should be skipped on decode when skip_special=true.
  std::vector<bool> is_special;

  int32_t vocab_size = 0;
  int32_t pad_id = -1;
  int32_t bos_id = -1;
  int32_t eos_id = -1;
  int32_t unk_id = -1;

  // Reverse of byte_to_unicode for decode.
  std::unordered_map<uint32_t, uint8_t> uc_to_byte;

  Impl() {
    const auto& tbl = byte_to_unicode();
    uc_to_byte.reserve(256);
    for (int b = 0; b < 256; ++b) {
      uc_to_byte[tbl[b]] = static_cast<uint8_t>(b);
    }
  }

  void load(const HfBpeTokenizerOptions& options);

  std::vector<int32_t> encode_text_segment(const std::string& text) const;
  std::vector<int32_t> bpe_encode_pretoken(const std::string& utf8_chunk) const;
  std::string decode_ids(const std::vector<int32_t>& ids, bool skip_special) const;
};

void HfBpeTokenizer::Impl::load(const HfBpeTokenizerOptions& options) {
  std::ifstream in(options.tokenizer_json_path);
  if (!in.is_open()) {
    throw std::runtime_error("HfBpeTokenizer: failed to open " + options.tokenizer_json_path);
  }
  json doc;
  in >> doc;

  if (!doc.contains("model")) {
    throw std::runtime_error("HfBpeTokenizer: tokenizer.json missing 'model' field");
  }
  const auto& model = doc["model"];
  std::string model_type = model.value("type", std::string{});
  if (model_type != "BPE") {
    throw std::runtime_error(
        "HfBpeTokenizer: unsupported model type '" + model_type +
        "' (only 'BPE' is supported; SentencePiece tokenizers need a different backend)");
  }

  // vocab.
  if (!model.contains("vocab") || !model["vocab"].is_object()) {
    throw std::runtime_error("HfBpeTokenizer: model.vocab missing or not an object");
  }
  vocab.reserve(model["vocab"].size() + 64);
  for (const auto& [key, value] : model["vocab"].items()) {
    if (!value.is_number_integer()) {
      throw std::runtime_error("HfBpeTokenizer: vocab value for '" + key + "' is not an integer");
    }
    int32_t id = value.get<int32_t>();
    vocab.emplace(key, id);
    if (id + 1 > vocab_size) vocab_size = id + 1;
  }

  // merges.
  if (model.contains("merges") && model["merges"].is_array()) {
    merge_ranks.reserve(model["merges"].size());
    int32_t rank = 0;
    for (const auto& m : model["merges"]) {
      // Newer tokenizer.json files store merges as ["a", "b"] arrays;
      // older ones as "a b" strings. Support both.
      std::string a, b;
      if (m.is_string()) {
        std::string s = m.get<std::string>();
        auto sp = s.find(' ');
        if (sp == std::string::npos) {
          throw std::runtime_error("HfBpeTokenizer: malformed merge string '" + s + "'");
        }
        a = s.substr(0, sp);
        b = s.substr(sp + 1);
      } else if (m.is_array() && m.size() == 2 && m[0].is_string() && m[1].is_string()) {
        a = m[0].get<std::string>();
        b = m[1].get<std::string>();
      } else {
        throw std::runtime_error("HfBpeTokenizer: unsupported merge entry shape");
      }
      merge_ranks.emplace(std::make_pair(std::move(a), std::move(b)), rank++);
    }
  }

  // added_tokens (special tokens).
  if (doc.contains("added_tokens") && doc["added_tokens"].is_array()) {
    for (const auto& tok : doc["added_tokens"]) {
      if (!tok.contains("id") || !tok.contains("content")) continue;
      int32_t id = tok["id"].get<int32_t>();
      std::string content = tok["content"].get<std::string>();
      bool special = tok.value("special", true);
      vocab[content] = id;
      added_tokens_sorted.emplace_back(content, id);
      if (id + 1 > vocab_size) vocab_size = id + 1;
      // Heuristics: pick up the standard names.
      if (special) {
        if (content == "<|pad|>" || content == "<pad>" || content == "[PAD]") {
          if (pad_id < 0) pad_id = id;
        }
        if (content == "<|endoftext|>" || content == "</s>" || content == "<|im_end|>" ||
            content == "<eos>" || content == "[EOS]") {
          if (eos_id < 0) eos_id = id;
        }
        if (content == "<s>" || content == "<bos>" || content == "[BOS]" ||
            content == "<|im_start|>") {
          if (bos_id < 0) bos_id = id;
        }
        if (content == "<unk>" || content == "[UNK]") {
          if (unk_id < 0) unk_id = id;
        }
      }
    }
    // Longest first so a longer special is matched before a shorter prefix.
    std::sort(added_tokens_sorted.begin(), added_tokens_sorted.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
  }

  // model.unk_token override.
  if (model.contains("unk_token") && model["unk_token"].is_string()) {
    auto it = vocab.find(model["unk_token"].get<std::string>());
    if (it != vocab.end()) unk_id = it->second;
  }

  // pad token from options or fallback.
  if (options.pad_id >= 0) {
    pad_id = options.pad_id;
  } else if (pad_id < 0) {
    auto it = vocab.find(options.pad_token_string);
    if (it != vocab.end()) {
      pad_id = it->second;
    } else if (eos_id >= 0) {
      pad_id = eos_id;  // Many models use eos as pad in inference.
    }
  }

  // Build reverse vocab.
  id_to_token.assign(static_cast<size_t>(vocab_size), std::string{});
  is_special.assign(static_cast<size_t>(vocab_size), false);
  for (const auto& [tok, id] : vocab) {
    if (id >= 0 && id < vocab_size) {
      id_to_token[static_cast<size_t>(id)] = tok;
    }
  }
  for (const auto& [content, id] : added_tokens_sorted) {
    if (id >= 0 && id < vocab_size) {
      is_special[static_cast<size_t>(id)] = true;
    }
  }

  // pre_tokenizer pattern. HF supports a Sequence of normalizer / split
  // patterns; we look for the first nested "Split" entry that has a
  // "Regex" pattern, otherwise fall back to a Qwen/GPT-2 style default.
  std::string pattern;
  if (doc.contains("pre_tokenizer")) {
    std::function<void(const json&)> walk = [&](const json& node) {
      if (!pattern.empty()) return;
      if (!node.is_object()) return;
      std::string ty = node.value("type", std::string{});
      if (ty == "Split" && node.contains("pattern") && node["pattern"].is_object()) {
        const auto& pat = node["pattern"];
        if (pat.contains("Regex") && pat["Regex"].is_string()) {
          pattern = pat["Regex"].get<std::string>();
          return;
        }
        if (pat.contains("String") && pat["String"].is_string()) {
          // Literal string split is rare here; fall through.
        }
      }
      if (ty == "Sequence" && node.contains("pretokenizers") && node["pretokenizers"].is_array()) {
        for (const auto& child : node["pretokenizers"]) walk(child);
      }
    };
    walk(doc["pre_tokenizer"]);
  }
  if (pattern.empty()) {
    pattern = kQwenLikePattern;
  }
  pre_tokenizer = std::make_unique<Pcre2Regex>(pattern);
}

std::vector<int32_t> HfBpeTokenizer::Impl::bpe_encode_pretoken(const std::string& utf8_chunk) const {
  if (utf8_chunk.empty()) return {};

  // Convert raw bytes -> byte-level unicode chars (each becomes its own
  // initial token in the BPE algorithm).
  std::vector<std::string> parts;
  parts.reserve(utf8_chunk.size());
  const auto& tbl = byte_to_unicode();
  for (unsigned char b : utf8_chunk) {
    std::string s;
    utf8_append(tbl[b], s);
    parts.push_back(std::move(s));
  }
  if (parts.size() == 1) {
    auto it = vocab.find(parts[0]);
    if (it != vocab.end()) return {it->second};
    if (unk_id >= 0) return {unk_id};
    return {};
  }

  // Iteratively merge the lowest-rank adjacent pair until none remain. O(N
  // * M) where N = parts.size() and M = number of merges performed; small
  // for typical pre-token chunks (tens of bytes).
  while (parts.size() > 1) {
    int32_t best_rank = std::numeric_limits<int32_t>::max();
    size_t best_idx = 0;
    bool found = false;
    for (size_t i = 0; i + 1 < parts.size(); ++i) {
      auto it = merge_ranks.find({parts[i], parts[i + 1]});
      if (it != merge_ranks.end() && it->second < best_rank) {
        best_rank = it->second;
        best_idx = i;
        found = true;
      }
    }
    if (!found) break;
    parts[best_idx] = parts[best_idx] + parts[best_idx + 1];
    parts.erase(parts.begin() + best_idx + 1);
  }

  std::vector<int32_t> ids;
  ids.reserve(parts.size());
  for (const auto& p : parts) {
    auto it = vocab.find(p);
    if (it != vocab.end()) {
      ids.push_back(it->second);
    } else if (unk_id >= 0) {
      ids.push_back(unk_id);
    }
    // Otherwise drop silently; for byte-level BPE every byte must exist in
    // the vocab as a single-char token, so this branch should not fire on
    // well-formed inputs.
  }
  return ids;
}

std::vector<int32_t> HfBpeTokenizer::Impl::encode_text_segment(const std::string& text) const {
  std::vector<int32_t> ids;
  std::vector<std::string> chunks;
  pre_tokenizer->find_all(text, chunks);
  for (const auto& chunk : chunks) {
    // Each chunk is raw UTF-8 from the input. bpe_encode_pretoken does the
    // byte_to_unicode mapping internally, so we pass raw bytes here.
    auto piece_ids = bpe_encode_pretoken(chunk);
    ids.insert(ids.end(), piece_ids.begin(), piece_ids.end());
  }
  return ids;
}

std::string HfBpeTokenizer::Impl::decode_ids(const std::vector<int32_t>& ids, bool skip_special) const {
  // Concatenate token strings, then reverse byte_to_unicode mapping to get
  // back the original bytes.
  std::string concat;
  concat.reserve(ids.size() * 4);
  for (int32_t id : ids) {
    if (id < 0 || id >= vocab_size) continue;
    if (skip_special && is_special[static_cast<size_t>(id)]) continue;
    concat += id_to_token[static_cast<size_t>(id)];
  }
  std::vector<uint32_t> codepoints;
  codepoints.reserve(concat.size());
  utf8_decode(concat, codepoints);
  std::string out;
  out.reserve(codepoints.size());
  for (uint32_t cp : codepoints) {
    auto it = uc_to_byte.find(cp);
    if (it != uc_to_byte.end()) {
      out.push_back(static_cast<char>(it->second));
    } else {
      // Codepoint outside the byte_to_unicode range — preserve as UTF-8 so
      // the caller still sees something meaningful (e.g. a special token's
      // raw content). This happens when skip_special=false and a special
      // token's content has non-byte-encoded characters.
      utf8_append(cp, out);
    }
  }
  return out;
}

HfBpeTokenizer::HfBpeTokenizer(const HfBpeTokenizerOptions& options)
    : impl_(std::make_unique<Impl>()) {
  impl_->load(options);
}

HfBpeTokenizer::~HfBpeTokenizer() = default;

std::unique_ptr<HfBpeTokenizer> HfBpeTokenizer::from_model_dir(const std::string& model_dir) {
  HfBpeTokenizerOptions opts;
  opts.tokenizer_json_path = model_dir + "/tokenizer.json";
  return std::make_unique<HfBpeTokenizer>(opts);
}

std::vector<int32_t> HfBpeTokenizer::encode(const std::string& text) const {
  return encode(text, EncodeOptions{});
}

std::vector<int32_t> HfBpeTokenizer::encode(const std::string& text, const EncodeOptions& options) const {
  std::vector<int32_t> ids;
  if (options.add_bos && impl_->bos_id >= 0) {
    ids.push_back(impl_->bos_id);
  }

  // Special-token-first split: scan the text for any added_tokens content,
  // emit the literal id at each match, BPE-encode the gaps in between.
  size_t pos = 0;
  const std::string& s = text;
  while (pos < s.size()) {
    size_t best_start = std::string::npos;
    size_t best_len = 0;
    int32_t best_id = -1;
    for (const auto& [content, id] : impl_->added_tokens_sorted) {
      if (content.empty()) continue;
      auto found = s.find(content, pos);
      if (found != std::string::npos &&
          (best_start == std::string::npos || found < best_start ||
           (found == best_start && content.size() > best_len))) {
        best_start = found;
        best_len = content.size();
        best_id = id;
      }
    }
    if (best_start == std::string::npos) {
      auto seg_ids = impl_->encode_text_segment(s.substr(pos));
      ids.insert(ids.end(), seg_ids.begin(), seg_ids.end());
      break;
    }
    if (best_start > pos) {
      auto seg_ids = impl_->encode_text_segment(s.substr(pos, best_start - pos));
      ids.insert(ids.end(), seg_ids.begin(), seg_ids.end());
    }
    ids.push_back(best_id);
    pos = best_start + best_len;
  }

  if (options.add_eos && impl_->eos_id >= 0) {
    if (options.max_tokens <= 0 || static_cast<int32_t>(ids.size()) < options.max_tokens) {
      ids.push_back(impl_->eos_id);
    }
  }
  if (options.max_tokens > 0 && static_cast<int32_t>(ids.size()) > options.max_tokens) {
    ids.resize(static_cast<size_t>(options.max_tokens));
  }
  return ids;
}

std::string HfBpeTokenizer::decode(const std::vector<int32_t>& ids, bool skip_special) const {
  return impl_->decode_ids(ids, skip_special);
}

int32_t HfBpeTokenizer::vocab_size() const { return impl_->vocab_size; }
int32_t HfBpeTokenizer::pad_id() const { return impl_->pad_id; }
int32_t HfBpeTokenizer::bos_id() const { return impl_->bos_id; }
int32_t HfBpeTokenizer::eos_id() const { return impl_->eos_id; }
int32_t HfBpeTokenizer::unk_id() const { return impl_->unk_id; }

}  // namespace cverl::text
