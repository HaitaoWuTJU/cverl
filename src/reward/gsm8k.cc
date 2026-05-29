#include "cverl/reward/gsm8k.h"

#include <cctype>
#include <stdexcept>
#include <string>

namespace cverl::reward {

namespace {

// "Number" matches the same character class as verl's regex
// (\\-?[0-9\\.\\,]+): an optional leading minus followed by digits, dots,
// and commas. Whitespace, $ and other punctuation are not part of a number.
bool is_number_char(char c) {
  return std::isdigit(static_cast<unsigned char>(c)) || c == '.' || c == ',' || c == '-';
}

std::string clip_tail(const std::string& s, size_t max_len) {
  if (s.size() <= max_len) {
    return s;
  }
  return s.substr(s.size() - max_len);
}

std::string strip_format_chars(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c != ',' && c != '$') {
      out.push_back(c);
    }
  }
  return out;
}

// Extract every (\\-?[0-9\\.\\,]+) match from `text`, in order.
std::vector<std::string> find_number_tokens(const std::string& text) {
  std::vector<std::string> out;
  const size_t n = text.size();
  size_t i = 0;
  while (i < n) {
    if (is_number_char(text[i])) {
      size_t start = i;
      // The verl regex allows a leading '-'; only treat it as part of a
      // number when it is actually at the start of a numeric run.
      while (i < n && is_number_char(text[i])) {
        ++i;
      }
      out.emplace_back(text.substr(start, i - start));
    } else {
      ++i;
    }
  }
  return out;
}

// Strict extractor: walk the string and capture every "#### <number>" pair.
// Returns the trailing one if any.
std::optional<std::string> extract_strict(const std::string& text) {
  std::optional<std::string> last;
  const std::string marker = "#### ";
  size_t pos = 0;
  while (true) {
    size_t hit = text.find(marker, pos);
    if (hit == std::string::npos) {
      break;
    }
    size_t start = hit + marker.size();
    size_t end = start;
    while (end < text.size() && is_number_char(text[end])) {
      ++end;
    }
    if (end > start) {
      last = text.substr(start, end - start);
    }
    pos = end;
  }
  return last;
}

}  // namespace

std::optional<std::string> extract_gsm8k_answer(const std::string& solution,
                                                const Gsm8kRewardOptions& options) {
  std::string clipped = clip_tail(solution, options.solution_clip_chars);
  if (options.method == Gsm8kExtractionMethod::Strict) {
    auto raw = extract_strict(clipped);
    if (!raw.has_value()) {
      return std::nullopt;
    }
    return strip_format_chars(*raw);
  }
  // Flexible: pick the last numeric token that is not "" or ".".
  auto tokens = find_number_tokens(clipped);
  if (tokens.empty()) {
    return std::nullopt;
  }
  for (auto it = tokens.rbegin(); it != tokens.rend(); ++it) {
    if (!it->empty() && *it != ".") {
      return *it;
    }
  }
  return std::nullopt;
}

std::string normalize_gsm8k_ground_truth(const std::string& ground_truth) {
  // Many GSM8K dumps store the gold as either bare digits or the full
  // explanation ending in "#### <number>". Try strict extraction first,
  // then fall back to the trimmed string with formatting chars removed.
  Gsm8kRewardOptions opts;
  opts.method = Gsm8kExtractionMethod::Strict;
  auto extracted = extract_gsm8k_answer(ground_truth, opts);
  if (extracted.has_value()) {
    return *extracted;
  }
  // Strip whitespace and formatting characters.
  std::string out;
  out.reserve(ground_truth.size());
  for (char c : ground_truth) {
    if (std::isspace(static_cast<unsigned char>(c))) {
      continue;
    }
    if (c == ',' || c == '$') {
      continue;
    }
    out.push_back(c);
  }
  return out;
}

double compute_gsm8k_reward(const std::string& solution,
                            const std::string& ground_truth,
                            const Gsm8kRewardOptions& options) {
  auto answer = extract_gsm8k_answer(solution, options);
  if (!answer.has_value()) {
    return options.no_answer_score;
  }
  std::string gold = normalize_gsm8k_ground_truth(ground_truth);
  if (*answer == gold) {
    return options.correct_score;
  }
  return options.format_score;
}

std::vector<double> compute_gsm8k_rewards(const std::vector<std::string>& solutions,
                                          const std::vector<std::string>& ground_truths,
                                          const Gsm8kRewardOptions& options) {
  if (solutions.size() != ground_truths.size()) {
    throw std::invalid_argument("compute_gsm8k_rewards: size mismatch");
  }
  std::vector<double> rewards(solutions.size());
  for (size_t i = 0; i < solutions.size(); ++i) {
    rewards[i] = compute_gsm8k_reward(solutions[i], ground_truths[i], options);
  }
  return rewards;
}

}  // namespace cverl::reward
