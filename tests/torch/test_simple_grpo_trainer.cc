#include "cverl/torch/simple_grpo_trainer.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
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

  const std::filesystem::path ckpt_prefix =
      std::filesystem::temp_directory_path() / "cverl_simple_grpo_trainer_test_ckpt";
  trainer.save_checkpoint(ckpt_prefix.string());

  cverl::torch_backend::SimpleGrpoTrainer restored(config);
  restored.load_checkpoint(ckpt_prefix.string());
  if (restored.current_step() != config.steps) {
    std::cerr << "restored checkpoint step mismatch\n";
    return 1;
  }
  auto resumed = restored.train_step();
  if (resumed.step != config.steps + 1 || !std::isfinite(resumed.loss)) {
    std::cerr << "failed to resume from checkpoint\n";
    return 1;
  }

  std::filesystem::remove(ckpt_prefix.string() + ".model.pt");
  std::filesystem::remove(ckpt_prefix.string() + ".optim.pt");
  std::filesystem::remove(ckpt_prefix.string() + ".meta.pt");

  std::cout << "simple GRPO trainer smoke test passed\n";
  return 0;
}
