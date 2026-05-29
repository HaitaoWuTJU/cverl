#include "cverl/data/hf_dataset.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::string arg_str(int argc, char** argv, const std::string& flag, const std::string& fallback = {}) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == flag) {
      return argv[i + 1];
    }
  }
  return fallback;
}

int64_t arg_i64(int argc, char** argv, const std::string& flag, int64_t fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == flag) {
      return std::stoll(argv[i + 1]);
    }
  }
  return fallback;
}

bool has_flag(int argc, char** argv, const std::string& flag) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == flag) {
      return true;
    }
  }
  return false;
}

void print_usage() {
  std::cerr
      << "usage: hf_dataset_download DATASET_ID --output-file PATH [options]\n"
      << "\n"
      << "options:\n"
      << "  --name NAME                  Dataset config, e.g. main for gsm8k\n"
      << "  --split SPLIT                Dataset split, default train\n"
      << "  --prompt-field FIELD         Source prompt field, default question\n"
      << "  --answer-field FIELD         Source answer field, default answer\n"
      << "  --max-examples N             Limit rows written\n"
      << "  --cache-dir DIR              Hugging Face datasets cache dir\n"
      << "  --token TOKEN                Hugging Face token\n"
      << "  --python PATH                Python interpreter with datasets installed\n"
      << "  --trust-remote-code          Pass trust_remote_code to datasets\n"
      << "  --local-files-only           Intended for HF_DATASETS_OFFLINE=1 runs\n"
      << "  --print-samples N            Print first N loaded JSONL rows\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || has_flag(argc, argv, "--help")) {
      print_usage();
      return argc < 2 ? 1 : 0;
    }

    cverl::data::HfDatasetOptions options;
    options.dataset_id = argv[1];
    options.name = arg_str(argc, argv, "--name");
    options.split = arg_str(argc, argv, "--split", "train");
    options.prompt_field = arg_str(argc, argv, "--prompt-field", "question");
    options.answer_field = arg_str(argc, argv, "--answer-field", "answer");
    options.output_file = arg_str(argc, argv, "--output-file");
    options.cache_dir = arg_str(argc, argv, "--cache-dir");
    options.token = arg_str(argc, argv, "--token");
    options.python_executable = arg_str(argc, argv, "--python", "python3");
    options.max_examples = arg_i64(argc, argv, "--max-examples", -1);
    options.local_files_only = has_flag(argc, argv, "--local-files-only");
    options.trust_remote_code = has_flag(argc, argv, "--trust-remote-code");

    auto result = cverl::data::download_hf_dataset(options);
    std::cout << result.output;
    if (!result.ok) {
      std::cerr << "HF dataset download failed with exit code " << result.exit_code << "\n";
      return result.exit_code == 0 ? 1 : result.exit_code;
    }

    int64_t print_samples = arg_i64(argc, argv, "--print-samples", 0);
    if (print_samples > 0) {
      cverl::data::JsonlDatasetOptions load_options;
      load_options.path = result.output_file;
      load_options.max_examples = print_samples;
      auto examples = cverl::data::load_prompt_answer_jsonl(load_options);
      for (size_t i = 0; i < examples.size(); ++i) {
        std::cout << "sample[" << i << "].prompt=" << examples[i].prompt << "\n";
        std::cout << "sample[" << i << "].answer=" << examples[i].answer << "\n";
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "hf_dataset_download failed: " << e.what() << "\n";
    return 1;
  }
}
