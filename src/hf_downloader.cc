#include "cverl/hf_downloader.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>

#ifndef CVERL_HF_DOWNLOAD_SCRIPT
#define CVERL_HF_DOWNLOAD_SCRIPT "tools/hf_snapshot_download.py"
#endif

namespace cverl {
namespace {

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

void append_vec(std::ostringstream& cmd, const std::string& name, const std::vector<std::string>& values) {
  for (const auto& value : values) {
    append_arg(cmd, name, value);
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

std::string parse_local_dir(const std::string& output) {
  const std::string marker = "CVERL_HF_LOCAL_DIR=";
  size_t pos = output.rfind(marker);
  if (pos == std::string::npos) {
    return {};
  }
  pos += marker.size();
  size_t end = output.find('\n', pos);
  return output.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
}

}  // namespace

HfDownloadResult download_hf_snapshot(const HfDownloadOptions& options) {
  if (options.repo_id.empty()) {
    throw std::invalid_argument("repo_id is required");
  }

  std::ostringstream cmd;
  cmd << shell_quote(options.python_executable) << " " << shell_quote(CVERL_HF_DOWNLOAD_SCRIPT);
  append_arg(cmd, "--repo-id", options.repo_id);
  append_arg(cmd, "--revision", options.revision);
  append_arg(cmd, "--local-dir", options.local_dir);
  append_arg(cmd, "--cache-dir", options.cache_dir);
  append_arg(cmd, "--token", options.token);
  append_vec(cmd, "--allow-pattern", options.allow_patterns);
  append_vec(cmd, "--ignore-pattern", options.ignore_patterns);
  if (options.local_files_only) {
    cmd << " --local-files-only";
  }
  if (options.dry_run) {
    cmd << " --dry-run";
  }
  cmd << " 2>&1";

  HfDownloadResult result;
  std::array<char, 4096> buffer{};
  FILE* pipe = popen(cmd.str().c_str(), "r");
  if (pipe == nullptr) {
    result.output = "failed to start downloader process";
    return result;
  }
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    result.output += buffer.data();
  }
  result.exit_code = decode_status(pclose(pipe));
  result.ok = result.exit_code == 0;
  result.local_dir = parse_local_dir(result.output);
  if (result.local_dir.empty()) {
    result.local_dir = options.local_dir;
  }
  return result;
}

}  // namespace cverl
