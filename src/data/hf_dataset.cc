#include "cverl/data/hf_dataset.h"

#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unordered_map>

#ifndef CVERL_HF_DATASET_SCRIPT
#define CVERL_HF_DATASET_SCRIPT "tools/hf_dataset_download.py"
#endif

namespace cverl::data {
namespace {

class JsonObjectParser {
 public:
  explicit JsonObjectParser(std::string text) : text_(std::move(text)) {}

  std::unordered_map<std::string, std::string> parse_string_object() {
    std::unordered_map<std::string, std::string> out;
    skip_ws();
    expect('{');
    skip_ws();
    if (consume('}')) {
      return out;
    }
    while (true) {
      skip_ws();
      std::string key = parse_string();
      skip_ws();
      expect(':');
      skip_ws();
      out.emplace(std::move(key), parse_value_as_string());
      skip_ws();
      if (consume('}')) {
        break;
      }
      expect(',');
    }
    return out;
  }

 private:
  void skip_ws() {
    while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
  }

  bool consume(char expected) {
    if (pos_ < text_.size() && text_[pos_] == expected) {
      ++pos_;
      return true;
    }
    return false;
  }

  void expect(char expected) {
    if (!consume(expected)) {
      throw std::invalid_argument("invalid JSON object");
    }
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (pos_ < text_.size()) {
      char ch = text_[pos_++];
      if (ch == '"') {
        return out;
      }
      if (ch != '\\') {
        out.push_back(ch);
        continue;
      }
      if (pos_ >= text_.size()) {
        throw std::invalid_argument("unterminated JSON escape");
      }
      char esc = text_[pos_++];
      switch (esc) {
        case '"':
        case '\\':
        case '/':
          out.push_back(esc);
          break;
        case 'b':
          out.push_back('\b');
          break;
        case 'f':
          out.push_back('\f');
          break;
        case 'n':
          out.push_back('\n');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 't':
          out.push_back('\t');
          break;
        case 'u':
          out += parse_unicode_escape_utf8();
          break;
        default:
          throw std::invalid_argument("unsupported JSON escape");
      }
    }
    throw std::invalid_argument("unterminated JSON string");
  }

  std::string parse_unicode_escape_utf8() {
    if (pos_ + 4 > text_.size()) {
      throw std::invalid_argument("short JSON unicode escape");
    }
    uint32_t code = 0;
    for (int i = 0; i < 4; ++i) {
      char ch = text_[pos_++];
      code <<= 4;
      if (ch >= '0' && ch <= '9') {
        code += static_cast<uint32_t>(ch - '0');
      } else if (ch >= 'a' && ch <= 'f') {
        code += static_cast<uint32_t>(ch - 'a' + 10);
      } else if (ch >= 'A' && ch <= 'F') {
        code += static_cast<uint32_t>(ch - 'A' + 10);
      } else {
        throw std::invalid_argument("invalid JSON unicode escape");
      }
    }
    if (code <= 0x7F) {
      return std::string(1, static_cast<char>(code));
    }
    if (code <= 0x7FF) {
      return std::string{static_cast<char>(0xC0 | (code >> 6)), static_cast<char>(0x80 | (code & 0x3F))};
    }
    return std::string{
        static_cast<char>(0xE0 | (code >> 12)),
        static_cast<char>(0x80 | ((code >> 6) & 0x3F)),
        static_cast<char>(0x80 | (code & 0x3F))};
  }

  std::string parse_value_as_string() {
    if (pos_ < text_.size() && text_[pos_] == '"') {
      return parse_string();
    }
    size_t begin = pos_;
    int depth = 0;
    while (pos_ < text_.size()) {
      char ch = text_[pos_];
      if (ch == '[' || ch == '{') {
        ++depth;
      } else if (ch == ']' || ch == '}') {
        if (depth == 0) {
          break;
        }
        --depth;
      } else if (ch == ',' && depth == 0) {
        break;
      }
      ++pos_;
    }
    size_t end = pos_;
    while (end > begin && std::isspace(static_cast<unsigned char>(text_[end - 1]))) {
      --end;
    }
    return text_.substr(begin, end - begin);
  }

  std::string text_;
  size_t pos_ = 0;
};

std::string shell_quote(const std::string& value) {
  std::string out = "'";
  for (char ch : value) {
    if (ch == '\'') {
      out += "'\\''";
    } else {
      out += ch;
    }
  }
  out += "'";
  return out;
}

void append_arg(std::ostringstream& cmd, const std::string& name, const std::string& value) {
  if (!value.empty()) {
    cmd << " " << name << " " << shell_quote(value);
  }
}

int decode_status(int status) {
  if (status == -1) {
    return -1;
  }
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  if (WIFSIGNALED(status)) {
    return 128 + WTERMSIG(status);
  }
  return status;
}

std::string parse_marker_string(const std::string& output, const std::string& marker) {
  size_t pos = output.rfind(marker);
  if (pos == std::string::npos) {
    return {};
  }
  pos += marker.size();
  size_t end = output.find('\n', pos);
  return output.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
}

int64_t parse_marker_i64(const std::string& output, const std::string& marker) {
  std::string value = parse_marker_string(output, marker);
  if (value.empty()) {
    return 0;
  }
  return std::stoll(value);
}

}  // namespace

std::vector<PromptAnswerExample> load_prompt_answer_jsonl(const JsonlDatasetOptions& options) {
  if (options.path.empty()) {
    throw std::invalid_argument("dataset path is required");
  }
  if (options.prompt_field.empty() || options.answer_field.empty()) {
    throw std::invalid_argument("prompt_field and answer_field are required");
  }

  std::ifstream in(options.path);
  if (!in) {
    throw std::runtime_error("failed to open dataset JSONL: " + options.path);
  }

  std::vector<PromptAnswerExample> examples;
  std::string line;
  int64_t line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    if (line.empty()) {
      continue;
    }
    auto object = JsonObjectParser(line).parse_string_object();
    auto prompt_it = object.find(options.prompt_field);
    auto answer_it = object.find(options.answer_field);
    if (prompt_it == object.end() || answer_it == object.end()) {
      throw std::runtime_error("dataset JSONL missing required field at line " + std::to_string(line_no));
    }
    examples.push_back(PromptAnswerExample{prompt_it->second, answer_it->second});
    if (options.max_examples >= 0 && static_cast<int64_t>(examples.size()) >= options.max_examples) {
      break;
    }
  }
  return examples;
}

HfDatasetResult download_hf_dataset(const HfDatasetOptions& options) {
  if (options.dataset_id.empty()) {
    throw std::invalid_argument("dataset_id is required");
  }
  if (options.output_file.empty()) {
    throw std::invalid_argument("output_file is required");
  }

  std::ostringstream cmd;
  cmd << shell_quote(options.python_executable) << " " << shell_quote(CVERL_HF_DATASET_SCRIPT);
  append_arg(cmd, "--dataset", options.dataset_id);
  append_arg(cmd, "--name", options.name);
  append_arg(cmd, "--split", options.split);
  append_arg(cmd, "--prompt-field", options.prompt_field);
  append_arg(cmd, "--answer-field", options.answer_field);
  append_arg(cmd, "--output-file", options.output_file);
  append_arg(cmd, "--cache-dir", options.cache_dir);
  append_arg(cmd, "--token", options.token);
  if (options.max_examples >= 0) {
    cmd << " --max-examples " << options.max_examples;
  }
  if (options.local_files_only) {
    cmd << " --local-files-only";
  }
  if (options.trust_remote_code) {
    cmd << " --trust-remote-code";
  }
  cmd << " 2>&1";

  HfDatasetResult result;
  std::array<char, 4096> buffer{};
  FILE* pipe = popen(cmd.str().c_str(), "r");
  if (pipe == nullptr) {
    result.output = "failed to start HF dataset process";
    return result;
  }
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    result.output += buffer.data();
  }
  result.exit_code = decode_status(pclose(pipe));
  result.ok = result.exit_code == 0;
  result.output_file = parse_marker_string(result.output, "CVERL_HF_DATASET_FILE=");
  if (result.output_file.empty()) {
    result.output_file = options.output_file;
  }
  result.rows = parse_marker_i64(result.output, "CVERL_HF_DATASET_ROWS=");
  return result;
}

}  // namespace cverl::data
