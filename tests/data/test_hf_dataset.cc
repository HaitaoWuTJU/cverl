#include "cverl/data/hf_dataset.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool cond, const char* msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

template <typename Fn>
void require_throws(Fn&& fn, const char* msg) {
  bool failed = false;
  try {
    fn();
  } catch (const std::exception&) {
    failed = true;
  }
  require(failed, msg);
}

void test_jsonl_reader() {
  auto path = std::filesystem::temp_directory_path() / "cverl_dataset_reader_test.jsonl";
  {
    std::ofstream out(path);
    out << "{\"prompt\":\"What is 2+2?\",\"answer\":\"4\",\"source\":\"gsm8k\"}\n";
    out << "{\"prompt\":\"line\\nwith newline\",\"answer\":\"escaped \\\"quote\\\"\"}\n";
    out << "{\"prompt\":\"unicode \\u03c0\",\"answer\":\"3.14\"}\n";
  }

  cverl::data::JsonlDatasetOptions options;
  options.path = path.string();
  options.max_examples = 2;
  auto examples = cverl::data::load_prompt_answer_jsonl(options);
  require(examples.size() == 2, "max_examples should limit rows");
  require(examples[0].prompt == "What is 2+2?", "first prompt");
  require(examples[0].answer == "4", "first answer");
  require(examples[1].prompt == "line\nwith newline", "escaped newline");
  require(examples[1].answer == "escaped \"quote\"", "escaped quote");

  options.max_examples = -1;
  examples = cverl::data::load_prompt_answer_jsonl(options);
  require(examples.size() == 3, "all rows");
  require(examples[2].prompt == "unicode \xcf\x80", "unicode escape");

  std::filesystem::remove(path);
}

void test_custom_fields_and_errors() {
  auto path = std::filesystem::temp_directory_path() / "cverl_dataset_custom_fields_test.jsonl";
  {
    std::ofstream out(path);
    out << "{\"question\":\"q\",\"final\":\"a\"}\n";
  }

  cverl::data::JsonlDatasetOptions options;
  options.path = path.string();
  options.prompt_field = "question";
  options.answer_field = "final";
  auto examples = cverl::data::load_prompt_answer_jsonl(options);
  require(examples.size() == 1, "custom fields row count");
  require(examples[0].prompt == "q" && examples[0].answer == "a", "custom fields values");

  options.answer_field = "missing";
  require_throws([&]() { (void)cverl::data::load_prompt_answer_jsonl(options); }, "missing field should fail");
  std::filesystem::remove(path);

  options.path.clear();
  require_throws([&]() { (void)cverl::data::load_prompt_answer_jsonl(options); }, "empty path should fail");
}

void test_downloader_validation() {
  cverl::data::HfDatasetOptions options;
  options.output_file = "/tmp/unused.jsonl";
  require_throws([&]() { (void)cverl::data::download_hf_dataset(options); }, "dataset id required");
  options.dataset_id = "gsm8k";
  options.output_file.clear();
  require_throws([&]() { (void)cverl::data::download_hf_dataset(options); }, "output file required");
}

}  // namespace

int main() {
  try {
    test_jsonl_reader();
    test_custom_fields_and_errors();
    test_downloader_validation();
  } catch (const std::exception& e) {
    std::cerr << "test_hf_dataset failed: " << e.what() << "\n";
    return 1;
  }
  std::cout << "HF dataset interface tests passed\n";
  return 0;
}
