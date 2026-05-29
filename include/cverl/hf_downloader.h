#pragma once

#include <string>
#include <vector>

namespace cverl {

struct HfDownloadOptions {
  std::string repo_id;
  std::string revision;
  std::string local_dir;
  std::string cache_dir;
  std::string token;
  std::string python_executable = "python3";
  std::vector<std::string> allow_patterns;
  std::vector<std::string> ignore_patterns;
  bool local_files_only = false;
  bool dry_run = false;
};

struct HfDownloadResult {
  bool ok = false;
  int exit_code = -1;
  std::string local_dir;
  std::string output;
};

HfDownloadResult download_hf_snapshot(const HfDownloadOptions& options);

}  // namespace cverl
