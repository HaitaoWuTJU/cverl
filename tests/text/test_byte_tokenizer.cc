#include "cverl/text/byte_tokenizer.h"

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
  using cverl::text::ByteTokenizer;
  try {
    ByteTokenizer tok;

    // Round-trip ASCII text without specials.
    {
      auto ids = tok.encode("hello", {});
      require(ids.size() == 5, "ascii encode length");
      require(ids[0] == ByteTokenizer::kByteOffset + 'h', "ascii first byte id");
      auto back = tok.decode(ids);
      require(back == "hello", "ascii round trip");
    }

    // Add BOS / EOS.
    {
      ByteTokenizer::EncodeOptions opts;
      opts.add_bos = true;
      opts.add_eos = true;
      auto ids = tok.encode("ok", opts);
      require(ids.size() == 4, "bos+eos length");
      require(ids.front() == ByteTokenizer::kBosId, "bos at start");
      require(ids.back() == ByteTokenizer::kEosId, "eos at end");
      // skip_special drops BOS/EOS.
      require(tok.decode(ids) == "ok", "decode skip special");
      // non-skip preserves them as text markers.
      auto verbose = tok.decode(ids, /*skip_special=*/false);
      require(verbose.find("<bos>") == 0, "verbose decode has bos");
      require(verbose.find("<eos>") != std::string::npos, "verbose decode has eos");
    }

    // Truncation honors max_tokens.
    {
      ByteTokenizer::EncodeOptions opts;
      opts.max_tokens = 3;
      auto ids = tok.encode("abcdef", opts);
      require(ids.size() == 3, "max_tokens truncates");
      require(tok.decode(ids) == "abc", "truncated decode");
    }

    // EOS only added when there's room.
    {
      ByteTokenizer::EncodeOptions opts;
      opts.add_eos = true;
      opts.max_tokens = 2;
      auto ids = tok.encode("abc", opts);
      require(ids.size() == 2, "max_tokens caps total");
      // No room left for EOS, so we get pure bytes.
      require(ids.back() == ByteTokenizer::kByteOffset + 'b', "no eos when full");
    }

    // Non-ASCII bytes survive the round trip byte-for-byte.
    {
      std::string utf8 = "héllo";  // 6 bytes
      auto ids = tok.encode(utf8, {});
      require(ids.size() == utf8.size(), "non-ascii length");
      require(tok.decode(ids) == utf8, "non-ascii round trip");
    }

    // is_byte_id only fires for byte ids.
    require(!ByteTokenizer::is_byte_id(ByteTokenizer::kPadId), "pad is not a byte");
    require(ByteTokenizer::is_byte_id(ByteTokenizer::kByteOffset), "byte 0 is byte");
    require(ByteTokenizer::is_byte_id(ByteTokenizer::kByteOffset + 255), "byte 255 is byte");

    std::cout << "byte tokenizer tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_byte_tokenizer failed: " << e.what() << "\n";
    return 1;
  }
}
