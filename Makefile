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

clean:
	rm -rf $(BUILD_DIR)
