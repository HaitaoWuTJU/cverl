#include "cverl/data/hf_dataset.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>

#include <nlohmann/json.hpp>

#ifndef CVERL_HF_DATASET_SCRIPT
#define CVERL_HF_DATASET_SCRIPT "tools/hf_dataset_download.py"
#endif

namespace cverl::data {
namespace {

using Json = nlohmann::json;

std::string json_value_to_string(const Json& value) {
  if (value.is_null()) {
    return {};
  }
  if (value.is_string()) {
    return value.get<std::string>();
  }
  return value.dump();
}

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
    Json object = Json::parse(line);
    if (!object.is_object()) {
      throw std::runtime_error("dataset JSONL row is not an object at line " + std::to_string(line_no));
    }
    if (!object.contains(options.prompt_field) || !object.contains(options.answer_field)) {
      throw std::runtime_error("dataset JSONL missing required field at line " + std::to_string(line_no));
    }
    examples.push_back(PromptAnswerExample{
        json_value_to_string(object.at(options.prompt_field)),
        json_value_to_string(object.at(options.answer_field)),
    });
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
