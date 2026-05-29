#pragma once

#include <optional>
#include <string>
#include <vector>

namespace cverl::reward {

enum class Gsm8kExtractionMethod {
  // Match "#### <number>" exactly, like the verl `strict` extractor.
  Strict,
  // Take the last number in the (clipped) tail of the string.
  Flexible,
};

struct Gsm8kRewardOptions {
  Gsm8kExtractionMethod method = Gsm8kExtractionMethod::Strict;
  // Reward returned when the answer is correct.
  double correct_score = 1.0;
  // Reward returned when an answer is parsed but does not match.
  double format_score = 0.0;
  // Reward returned when no answer can be extracted at all.
  double no_answer_score = 0.0;
  // verl trims to the last 300 chars because regex on long strings is slow
  // and the canonical answer always sits at the tail.
  size_t solution_clip_chars = 300;
};

// Extract the last `#### <number>` (Strict) or the last numeric token
// (Flexible) from `solution`. Returns std::nullopt when nothing matches.
std::optional<std::string> extract_gsm8k_answer(const std::string& solution,
                                                const Gsm8kRewardOptions& options = {});

// Normalize a gsm8k ground-truth answer. The dataset stores the gold answer
// either as the full chain-of-thought ending in "#### N" or just the number;
// this strips formatting characters so it can be compared against the
// extractor output.
std::string normalize_gsm8k_ground_truth(const std::string& ground_truth);

// Compute a single GSM8K reward.
double compute_gsm8k_reward(const std::string& solution,
                            const std::string& ground_truth,
                            const Gsm8kRewardOptions& options = {});

// Batch helper that scores `solutions[i]` against `ground_truths[i]` and
// returns one reward per pair.
std::vector<double> compute_gsm8k_rewards(const std::vector<std::string>& solutions,
                                          const std::vector<std::string>& ground_truths,
                                          const Gsm8kRewardOptions& options = {});

}  // namespace cverl::reward
