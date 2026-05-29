#include "cverl/reward/gsm8k.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool cond, const std::string& msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

void approx(double got, double want, const std::string& msg) {
  if (std::fabs(got - want) > 1e-9) {
    throw std::runtime_error(msg + " expected " + std::to_string(want) + " got " + std::to_string(got));
  }
}

}  // namespace

int main() {
  using cverl::reward::compute_gsm8k_reward;
  using cverl::reward::extract_gsm8k_answer;
  using cverl::reward::Gsm8kExtractionMethod;
  using cverl::reward::Gsm8kRewardOptions;
  using cverl::reward::normalize_gsm8k_ground_truth;

  try {
    // Strict extraction picks the trailing #### answer.
    {
      auto got = extract_gsm8k_answer("Reasoning... blah\n#### 18", {});
      require(got.has_value() && *got == "18", "strict simple");
    }

    // Strict extraction strips comma formatting (matches verl regex
    // capture group [-0-9.,] then .replace(",", "")).
    {
      Gsm8kRewardOptions opts;
      opts.method = Gsm8kExtractionMethod::Strict;
      auto got = extract_gsm8k_answer("text\n#### 1,234", opts);
      require(got.has_value() && *got == "1234", "strict comma strip");
    }

    // The dollar sign falls outside verl's capture class so it terminates
    // the match; the extractor then sees nothing after "#### ".
    {
      Gsm8kRewardOptions opts;
      opts.method = Gsm8kExtractionMethod::Strict;
      auto got = extract_gsm8k_answer("text\n#### $1,234", opts);
      require(!got.has_value(), "strict ignores $-prefixed answer like verl");
    }

    // When there are multiple #### markers, the last one wins.
    {
      auto got = extract_gsm8k_answer("#### 1\nmore\n#### 42", {});
      require(got.has_value() && *got == "42", "strict picks last");
    }

    // Strict returns nullopt when no marker is present.
    {
      auto got = extract_gsm8k_answer("the answer is 42", {});
      require(!got.has_value(), "strict requires marker");
    }

    // Flexible returns the trailing numeric token.
    {
      Gsm8kRewardOptions opts;
      opts.method = Gsm8kExtractionMethod::Flexible;
      auto got = extract_gsm8k_answer("step 1: blah. final answer is 42.", opts);
      require(got.has_value(), "flexible has answer");
      // The trailing '.' makes the last token "42." -> stripped of formatting later.
      // We accept either "42" or "42." since the verl flexible extractor returns "42."
      // and then equality is compared char-for-char. For our gold normalization
      // we strip trailing dots to be tolerant.
      require(*got == "42." || *got == "42", "flexible last number got=" + *got);
    }

    // Flexible skips standalone "." tokens.
    {
      Gsm8kRewardOptions opts;
      opts.method = Gsm8kExtractionMethod::Flexible;
      auto got = extract_gsm8k_answer("blah . . . 7", opts);
      require(got.has_value() && *got == "7", "flexible skips dots");
    }

    // verl clips long inputs to last 300 chars; an early #### marker is ignored.
    {
      std::string head(400, 'x');
      auto got = extract_gsm8k_answer("#### 1\n" + head + "\n#### 99", {});
      require(got.has_value() && *got == "99", "clip respects tail");
    }

    // Ground truth normalization: full chain-of-thought ends with "#### N".
    {
      auto gt = normalize_gsm8k_ground_truth("Lots of explanation.\n#### 18");
      require(gt == "18", "normalize from cot");
    }

    // Ground truth normalization: bare number with formatting.
    {
      auto gt = normalize_gsm8k_ground_truth(" 1,234 ");
      require(gt == "1234", "normalize bare number");
    }

    // End-to-end reward: correct answer -> 1.0.
    {
      double r = compute_gsm8k_reward("blah\n#### 18", "Lots of explanation.\n#### 18", {});
      approx(r, 1.0, "correct gives 1");
    }

    // End-to-end reward: parsed but wrong -> format_score.
    {
      Gsm8kRewardOptions opts;
      opts.format_score = 0.1;
      double r = compute_gsm8k_reward("blah\n#### 19", "Lots of explanation.\n#### 18", opts);
      approx(r, 0.1, "wrong gives format_score");
    }

    // End-to-end reward: missing answer -> no_answer_score.
    {
      Gsm8kRewardOptions opts;
      opts.no_answer_score = -0.5;
      double r = compute_gsm8k_reward("rambling without an answer", "#### 18", opts);
      approx(r, -0.5, "missing gives no_answer");
    }

    std::cout << "gsm8k reward tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_gsm8k_reward failed: " << e.what() << "\n";
    return 1;
  }
}
