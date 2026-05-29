#include "cverl/hf_downloader.h"

#include <iostream>
#include <string>

namespace {

void usage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " <repo-id> [--local-dir DIR] [--cache-dir DIR] [--revision REV]"
               " [--python PYTHON] [--token TOKEN] [--allow-pattern PAT] [--ignore-pattern PAT]"
               " [--local-files-only] [--dry-run]\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage(argv[0]);
    return 2;
  }

  cverl::HfDownloadOptions options;
  options.repo_id = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    auto need_value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << name << " requires a value\n";
        std::exit(2);
      }
      return argv[++i];
    };

    if (arg == "--local-dir") {
      options.local_dir = need_value("--local-dir");
    } else if (arg == "--cache-dir") {
      options.cache_dir = need_value("--cache-dir");
    } else if (arg == "--revision") {
      options.revision = need_value("--revision");
    } else if (arg == "--python") {
      options.python_executable = need_value("--python");
    } else if (arg == "--token") {
      options.token = need_value("--token");
    } else if (arg == "--allow-pattern") {
      options.allow_patterns.push_back(need_value("--allow-pattern"));
    } else if (arg == "--ignore-pattern") {
      options.ignore_patterns.push_back(need_value("--ignore-pattern"));
    } else if (arg == "--local-files-only") {
      options.local_files_only = true;
    } else if (arg == "--dry-run") {
      options.dry_run = true;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      usage(argv[0]);
      return 2;
    }
  }

  auto result = cverl::download_hf_snapshot(options);
  std::cout << result.output;
  if (!result.ok) {
    std::cerr << "hf download failed with exit code " << result.exit_code << "\n";
    return result.exit_code == 0 ? 1 : result.exit_code;
  }
  return 0;
}
