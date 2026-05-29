#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cverl::data {

struct PromptAnswerExample {
  std::string prompt;
  std::string answer;
};

struct JsonlDatasetOptions {
  std::string path;
  std::string prompt_field = "prompt";
  std::string answer_field = "answer";
  int64_t max_examples = -1;
};

struct HfDatasetOptions {
  std::string dataset_id;
  std::string name;
  std::string split = "train";
  std::string prompt_field = "question";
  std::string answer_field = "answer";
  std::string output_file;
  std::string cache_dir;
  std::string token;
  std::string python_executable = "python3";
  int64_t max_examples = -1;
  bool local_files_only = false;
  bool trust_remote_code = false;
};

struct HfDatasetResult {
  bool ok = false;
  int exit_code = -1;
  std::string output_file;
  int64_t rows = 0;
  std::string output;
};

std::vector<PromptAnswerExample> load_prompt_answer_jsonl(const JsonlDatasetOptions& options);
HfDatasetResult download_hf_dataset(const HfDatasetOptions& options);

}  // namespace cverl::data
