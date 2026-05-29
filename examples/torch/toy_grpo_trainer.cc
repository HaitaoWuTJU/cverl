#include "cverl/torch/simple_grpo_trainer.h"

#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
  cverl::torch_backend::SimpleGrpoTrainerConfig config;
  if (argc > 1) {
    config.steps = std::strtoll(argv[1], nullptr, 10);
  }
  if (argc > 2) {
    config.prompts_per_batch = std::strtoll(argv[2], nullptr, 10);
  }
  if (argc > 3) {
    config.responses_per_prompt = std::strtoll(argv[3], nullptr, 10);
  }

  cverl::torch_backend::SimpleGrpoTrainer trainer(config);
  auto metrics = trainer.train();

  std::cout << "step,loss,avg_reward,success_rate,ppo_kl,clipfrac\n";
  for (const auto& m : metrics) {
    std::cout << m.step << "," << m.loss << "," << m.avg_reward << "," << m.success_rate << "," << m.ppo_kl
              << "," << m.clipfrac << "\n";
  }
  return 0;
}
