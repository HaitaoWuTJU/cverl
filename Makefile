CMAKE ?= cmake
PYTHON ?= python3
BUILD_DIR ?= build
TORCH_CMAKE_PREFIX ?= $(shell $(PYTHON) -c 'import torch; print(torch.utils.cmake_prefix_path)')

.PHONY: all configure test clean

all: configure
	$(CMAKE) --build $(BUILD_DIR)

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_PREFIX_PATH="$(TORCH_CMAKE_PREFIX)"

test: all
	./$(BUILD_DIR)/test_core_algos_cpu
	./$(BUILD_DIR)/test_torch_backend
	./$(BUILD_DIR)/test_simple_grpo_trainer
	./$(BUILD_DIR)/test_hf_dataset
	./$(BUILD_DIR)/test_rollout_shared_memory
	./$(BUILD_DIR)/test_rollout_shared_memory_transport
	./$(BUILD_DIR)/test_rollout_transport
	./$(BUILD_DIR)/test_rollout_worker
	./$(BUILD_DIR)/test_gsm8k_reward
	./$(BUILD_DIR)/test_byte_tokenizer
	./$(BUILD_DIR)/test_hf_bpe_tokenizer
	./$(BUILD_DIR)/test_rollout_batch
	./$(BUILD_DIR)/test_gsm8k_grpo_trainer
	./$(BUILD_DIR)/test_reference_policy_kl
	./$(BUILD_DIR)/test_qwen3_5_causal_lm_policy
	./$(BUILD_DIR)/test_qwen3_5_grpo_step
	./$(BUILD_DIR)/test_distributed_topology
	./$(BUILD_DIR)/test_parallel_ops
	./$(BUILD_DIR)/test_weight_sync

clean:
	rm -rf $(BUILD_DIR)
