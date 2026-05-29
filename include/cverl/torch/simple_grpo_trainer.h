#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cverl::torch_backend {

struct SimpleGrpoTrainerConfig {
  int64_t steps = 32;
  int64_t prompts_per_batch = 8;
  int64_t responses_per_prompt = 4;
  int64_t seq_len = 8;
  int64_t action_dim = 8;
  int64_t hidden_dim = 64;
  int64_t ppo_epochs = 2;
  double learning_rate = 3.0e-3;
  double clip_ratio = 0.2;
  double entropy_coef = 0.0;
  uint64_t seed = 17;
};

struct SimpleGrpoTrainerMetrics {
  int64_t step = 0;
  double loss = 0.0;
  double avg_reward = 0.0;
  double success_rate = 0.0;
  double ppo_kl = 0.0;
  double clipfrac = 0.0;
};

class SimpleGrpoTrainer {
 public:
  explicit SimpleGrpoTrainer(SimpleGrpoTrainerConfig config);
  ~SimpleGrpoTrainer();

  SimpleGrpoTrainer(const SimpleGrpoTrainer&) = delete;
  SimpleGrpoTrainer& operator=(const SimpleGrpoTrainer&) = delete;
  SimpleGrpoTrainer(SimpleGrpoTrainer&&) noexcept;
  SimpleGrpoTrainer& operator=(SimpleGrpoTrainer&&) noexcept;

  SimpleGrpoTrainerMetrics train_step();
  std::vector<SimpleGrpoTrainerMetrics> train();

  void save_checkpoint(const std::string& prefix) const;
  void load_checkpoint(const std::string& prefix);
  int64_t current_step() const;

  const SimpleGrpoTrainerConfig& config() const { return config_; }

 private:
  struct Impl;

  SimpleGrpoTrainerConfig config_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cverl::torch_backend
