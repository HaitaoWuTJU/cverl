#include "cverl/model/hf_model_loader.h"

#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <unordered_map>

namespace cverl {
namespace {

std::string read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open " + path.string());
  }
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::string regex_string(const std::string& text, const std::string& pattern) {
  std::smatch match;
  if (std::regex_search(text, match, std::regex(pattern))) {
    return match[1].str();
  }
  return {};
}

int64_t regex_i64(const std::string& text, const std::string& pattern) {
  std::smatch match;
  if (std::regex_search(text, match, std::regex(pattern))) {
    return std::stoll(match[1].str());
  }
  return 0;
}

HfConfigSummary parse_config(const std::filesystem::path& path) {
  std::string text = read_file(path);
  HfConfigSummary cfg;
  cfg.model_type = regex_string(text, "\"model_type\"\\s*:\\s*\"([^\"]+)\"");
  cfg.architecture = regex_string(text, "\"architectures\"\\s*:\\s*\\[\\s*\"([^\"]+)\"");
  cfg.dtype = regex_string(text, "\"dtype\"\\s*:\\s*\"([^\"]+)\"");
  cfg.vocab_size = regex_i64(text, R"("vocab_size"\s*:\s*([0-9]+))");
  cfg.hidden_size = regex_i64(text, R"("hidden_size"\s*:\s*([0-9]+))");
  cfg.num_hidden_layers = regex_i64(text, R"("num_hidden_layers"\s*:\s*([0-9]+))");
  cfg.num_attention_heads = regex_i64(text, R"("num_attention_heads"\s*:\s*([0-9]+))");
  cfg.num_key_value_heads = regex_i64(text, R"("num_key_value_heads"\s*:\s*([0-9]+))");
  return cfg;
}

class JsonScanner {
 public:
  explicit JsonScanner(std::string_view text) : text_(text) {}

  bool eof() const { return pos_ >= text_.size(); }

  void skip_ws() {
    while (!eof() && std::isspace(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
  }

  bool consume(char expected) {
    skip_ws();
    if (eof() || text_[pos_] != expected) {
      return false;
    }
    ++pos_;
    return true;
  }

  void expect(char expected) {
    if (!consume(expected)) {
      throw std::runtime_error(std::string("expected JSON char ") + expected);
    }
  }

  std::string string() {
    skip_ws();
    expect('"');
    std::string out;
    while (!eof()) {
      char ch = text_[pos_++];
      if (ch == '"') {
        return out;
      }
      if (ch == '\\') {
        if (eof()) {
          throw std::runtime_error("unterminated JSON escape");
        }
        char escaped = text_[pos_++];
        switch (escaped) {
          case '"':
          case '\\':
          case '/':
            out.push_back(escaped);
            break;
          case 'n':
            out.push_back('\n');
            break;
          case 't':
            out.push_back('\t');
            break;
          default:
            out.push_back(escaped);
            break;
        }
      } else {
        out.push_back(ch);
      }
    }
    throw std::runtime_error("unterminated JSON string");
  }

  int64_t integer() {
    skip_ws();
    size_t begin = pos_;
    if (!eof() && text_[pos_] == '-') {
      ++pos_;
    }
    while (!eof() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
    if (begin == pos_) {
      throw std::runtime_error("expected JSON integer");
    }
    return std::stoll(std::string(text_.substr(begin, pos_ - begin)));
  }

  std::vector<int64_t> int_array() {
    std::vector<int64_t> out;
    expect('[');
    skip_ws();
    if (consume(']')) {
      return out;
    }
    while (true) {
      out.push_back(integer());
      if (consume(']')) {
        return out;
      }
      expect(',');
    }
  }

  void skip_value() {
    skip_ws();
    if (eof()) {
      throw std::runtime_error("unexpected JSON eof");
    }
    char ch = text_[pos_];
    if (ch == '"') {
      (void)string();
      return;
    }
    if (ch == '{') {
      expect('{');
      skip_ws();
      if (consume('}')) {
        return;
      }
      while (true) {
        (void)string();
        expect(':');
        skip_value();
        if (consume('}')) {
          return;
        }
        expect(',');
      }
    }
    if (ch == '[') {
      expect('[');
      skip_ws();
      if (consume(']')) {
        return;
      }
      while (true) {
        skip_value();
        if (consume(']')) {
          return;
        }
        expect(',');
      }
    }
    while (!eof() && std::string_view(",}]").find(text_[pos_]) == std::string_view::npos) {
      ++pos_;
    }
  }

 private:
  std::string_view text_;
  size_t pos_ = 0;
};

SafetensorInfo parse_tensor_object(JsonScanner& scanner, const std::string& name, const std::string& filename) {
  SafetensorInfo info;
  info.name = name;
  info.filename = filename;
  scanner.expect('{');
  scanner.skip_ws();
  if (scanner.consume('}')) {
    return info;
  }
  while (true) {
    std::string key = scanner.string();
    scanner.expect(':');
    if (key == "dtype") {
      info.dtype = scanner.string();
    } else if (key == "shape") {
      info.shape = scanner.int_array();
    } else if (key == "data_offsets") {
      std::vector<int64_t> offsets = scanner.int_array();
      if (offsets.size() != 2 || offsets[0] < 0 || offsets[1] < offsets[0]) {
        throw std::runtime_error("invalid safetensors offsets for " + name);
      }
      info.data_begin = static_cast<uint64_t>(offsets[0]);
      info.data_end = static_cast<uint64_t>(offsets[1]);
    } else {
      scanner.skip_value();
    }
    if (scanner.consume('}')) {
      return info;
    }
    scanner.expect(',');
  }
}

uint64_t read_u64_le(std::ifstream& in) {
  unsigned char bytes[8];
  in.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
  if (!in) {
    throw std::runtime_error("failed to read safetensors header length");
  }
  uint64_t value = 0;
  for (int i = 7; i >= 0; --i) {
    value = (value << 8) | bytes[i];
  }
  return value;
}

std::vector<SafetensorInfo> parse_safetensors_header(const std::filesystem::path& path, const std::string& filename) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open " + path.string());
  }
  uint64_t header_len = read_u64_le(in);
  std::string header(header_len, '\0');
  in.read(header.data(), static_cast<std::streamsize>(header.size()));
  if (!in) {
    throw std::runtime_error("failed to read safetensors header");
  }

  JsonScanner scanner(header);
  std::vector<SafetensorInfo> tensors;
  scanner.expect('{');
  scanner.skip_ws();
  if (scanner.consume('}')) {
    return tensors;
  }
  while (true) {
    std::string key = scanner.string();
    scanner.expect(':');
    if (key == "__metadata__") {
      scanner.skip_value();
    } else {
      tensors.push_back(parse_tensor_object(scanner, key, filename));
    }
    if (scanner.consume('}')) {
      return tensors;
    }
    scanner.expect(',');
  }
}

torch::ScalarType torch_dtype(const std::string& dtype) {
  if (dtype == "F32") {
    return torch::kFloat32;
  }
  if (dtype == "F16") {
    return torch::kFloat16;
  }
  if (dtype == "BF16") {
    return torch::kBFloat16;
  }
  if (dtype == "I64") {
    return torch::kInt64;
  }
  if (dtype == "I32") {
    return torch::kInt32;
  }
  if (dtype == "U8") {
    return torch::kUInt8;
  }
  throw std::runtime_error("unsupported safetensors dtype: " + dtype);
}

}  // namespace

HfModelLoader::HfModelLoader(std::string model_dir) : model_dir_(std::move(model_dir)) {}

void HfModelLoader::load_metadata() {
  std::filesystem::path root(model_dir_);
  config_ = parse_config(root / "config.json");
  tensors_.clear();
  tensor_index_.clear();

  for (const auto& entry : std::filesystem::directory_iterator(root)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    std::string filename = entry.path().filename().string();
    const std::string suffix = ".safetensors";
    if (filename.size() >= suffix.size() && filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) == 0) {
      auto tensors = parse_safetensors_header(entry.path(), filename);
      for (auto& tensor : tensors) {
        tensor_index_[tensor.name] = tensors_.size();
        tensors_.push_back(std::move(tensor));
      }
    }
  }
}

const SafetensorInfo* HfModelLoader::find_tensor(const std::string& name) const {
  auto it = tensor_index_.find(name);
  if (it == tensor_index_.end()) {
    return nullptr;
  }
  return &tensors_[it->second];
}

torch::Tensor HfModelLoader::load_tensor(const std::string& name) const {
  const SafetensorInfo* info = find_tensor(name);
  if (info == nullptr) {
    throw std::runtime_error("tensor not found: " + name);
  }

  std::filesystem::path path = std::filesystem::path(model_dir_) / info->filename;
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open " + path.string());
  }
  uint64_t header_len = read_u64_le(in);
  uint64_t absolute_begin = 8 + header_len + info->data_begin;
  uint64_t nbytes = info->data_end - info->data_begin;
  in.seekg(static_cast<std::streamoff>(absolute_begin), std::ios::beg);
  if (!in) {
    throw std::runtime_error("failed to seek tensor data: " + name);
  }

  torch::Tensor tensor = torch::empty(info->shape, torch_dtype(info->dtype));
  if (static_cast<uint64_t>(tensor.nbytes()) != nbytes) {
    throw std::runtime_error("tensor byte size mismatch: " + name);
  }
  in.read(static_cast<char*>(tensor.data_ptr()), static_cast<std::streamsize>(nbytes));
  if (!in) {
    throw std::runtime_error("failed to read tensor data: " + name);
  }
  return tensor;
}

std::unordered_map<std::string, torch::Tensor> HfModelLoader::load_all_tensors() const {
  std::unordered_map<std::string, torch::Tensor> out;
  out.reserve(tensors_.size());
  for (const auto& info : tensors_) {
    out.emplace(info.name, load_tensor(info.name));
  }
  return out;
}

}  // namespace cverl
