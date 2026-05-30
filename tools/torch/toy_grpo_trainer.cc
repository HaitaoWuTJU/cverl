#include "cverl/torch/simple_grpo_trainer.h"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  cverl::torch_backend::SimpleGrpoTrainerConfig config;
  std::string save_prefix;
  std::string load_prefix;
  if (argc > 1) {
    config.steps = std::strtoll(argv[1], nullptr, 10);
  }
  if (argc > 2) {
    config.prompts_per_batch = std::strtoll(argv[2], nullptr, 10);
  }
  if (argc > 3) {
    config.responses_per_prompt = std::strtoll(argv[3], nullptr, 10);
  }
  for (int i = 4; i + 1 < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--save-prefix") {
      save_prefix = argv[++i];
    } else if (arg == "--load-prefix") {
      load_prefix = argv[++i];
    }
  }

  cverl::torch_backend::SimpleGrpoTrainer trainer(config);
  if (!load_prefix.empty()) {
    trainer.load_checkpoint(load_prefix);
  }
  auto metrics = trainer.train();

  std::cout << "step,loss,avg_reward,success_rate,ppo_kl,clipfrac\n";
  for (const auto& m : metrics) {
    std::cout << m.step << "," << m.loss << "," << m.avg_reward << "," << m.success_rate << "," << m.ppo_kl
              << "," << m.clipfrac << "\n";
  }
  if (!save_prefix.empty()) {
    trainer.save_checkpoint(save_prefix);
  }
  return 0;
}
