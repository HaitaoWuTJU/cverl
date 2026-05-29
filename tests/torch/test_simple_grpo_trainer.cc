#include "cverl/torch/simple_grpo_trainer.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

int main() {
  cverl::torch_backend::SimpleGrpoTrainerConfig config;
  config.steps = 5;
  config.prompts_per_batch = 4;
  config.responses_per_prompt = 4;
  config.seq_len = 6;
  config.action_dim = 6;
  config.hidden_dim = 32;
  config.ppo_epochs = 1;
  config.seed = 29;

  cverl::torch_backend::SimpleGrpoTrainer trainer(config);
  auto metrics = trainer.train();
  if (metrics.size() != static_cast<size_t>(config.steps)) {
    std::cerr << "unexpected metrics size\n";
    return 1;
  }

  for (const auto& m : metrics) {
    if (!std::isfinite(m.loss) || !std::isfinite(m.avg_reward) || !std::isfinite(m.ppo_kl) ||
        !std::isfinite(m.clipfrac)) {
      std::cerr << "non-finite metric at step " << m.step << "\n";
      return 1;
    }
    if (m.avg_reward < 0.0 || m.avg_reward > 1.0 || m.success_rate < 0.0 || m.success_rate > 1.0) {
      std::cerr << "metric out of range at step " << m.step << "\n";
      return 1;
    }
  }

  std::cout << "simple GRPO trainer smoke test passed\n";
  return 0;
}
