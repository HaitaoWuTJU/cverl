cmake_minimum_required(VERSION 3.18)
project(cverl LANGUAGES CXX)

option(CVERL_BUILD_TESTS "Build cverl tests" ON)
option(CVERL_ENABLE_CUDA "Build CUDA kernels" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)

add_library(cverl STATIC
  src/status.cc
  src/hf_downloader.cc
  src/model/hf_model_loader.cc
  src/core_algos_c_api.cc
  src/torch/core_algos_torch.cc
  src/torch/simple_grpo_trainer.cc
)
target_include_directories(cverl PUBLIC include)
target_link_libraries(cverl PUBLIC "${TORCH_LIBRARIES}")
target_compile_options(cverl PRIVATE ${TORCH_CXX_FLAGS})
target_compile_definitions(cverl PRIVATE
  CVERL_HF_DOWNLOAD_SCRIPT="${CMAKE_CURRENT_SOURCE_DIR}/tools/hf_snapshot_download.py"
)

if (CVERL_ENABLE_CUDA)
  enable_language(CUDA)
  target_sources(cverl PRIVATE
    cuda/core_algos_cuda.cu
  )
endif()

if (CVERL_BUILD_TESTS)
  add_executable(test_core_algos_cpu tests/test_core_algos_cpu.cc)
  target_link_libraries(test_core_algos_cpu PRIVATE cverl)

  add_executable(compare_golden tools/compare_golden.cc)
  target_link_libraries(compare_golden PRIVATE cverl)

  add_executable(hf_download tools/hf_download.cc)
  target_link_libraries(hf_download PRIVATE cverl)

  add_executable(inspect_hf_model tools/model/inspect_hf_model.cc)
  target_link_libraries(inspect_hf_model PRIVATE cverl)

  add_executable(test_torch_backend tests/torch/test_torch_backend.cc)
  target_link_libraries(test_torch_backend PRIVATE cverl)
  target_compile_options(test_torch_backend PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(test_simple_grpo_trainer tests/torch/test_simple_grpo_trainer.cc)
  target_link_libraries(test_simple_grpo_trainer PRIVATE cverl)
  target_compile_options(test_simple_grpo_trainer PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(test_hf_model_loader tests/model/test_hf_model_loader.cc)
  target_link_libraries(test_hf_model_loader PRIVATE cverl)
  target_compile_options(test_hf_model_loader PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(minimal_ppo_step examples/torch/minimal_ppo_step.cc)
  target_link_libraries(minimal_ppo_step PRIVATE cverl)
  target_compile_options(minimal_ppo_step PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(toy_grpo_trainer examples/torch/toy_grpo_trainer.cc)
  target_link_libraries(toy_grpo_trainer PRIVATE cverl)
  target_compile_options(toy_grpo_trainer PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(hf_embedding_grpo_trainer examples/model/hf_embedding_grpo_trainer.cc)
  target_link_libraries(hf_embedding_grpo_trainer PRIVATE cverl)
  target_compile_options(hf_embedding_grpo_trainer PRIVATE ${TORCH_CXX_FLAGS})
endif()
