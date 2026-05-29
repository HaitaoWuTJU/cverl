cmake_minimum_required(VERSION 3.18)
project(cverl LANGUAGES CXX)

option(CVERL_BUILD_TESTS "Build cverl tests" ON)
option(CVERL_ENABLE_CUDA "Build CUDA kernels" OFF)
option(CVERL_ENABLE_NCCL "Build NCCL collectives" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)

find_package(simdjson CONFIG QUIET
  HINTS
    "$ENV{CONDA_PREFIX}"
    "$ENV{HOME}/miniconda3"
    "$ENV{HOME}/miniconda3/envs/verl-cpu"
)
if (NOT simdjson_FOUND)
  find_path(SIMDJSON_INCLUDE_DIR simdjson.h
    HINTS
      "$ENV{CONDA_PREFIX}/include"
      "$ENV{HOME}/miniconda3/include"
      "$ENV{HOME}/miniconda3/envs/verl-cpu/include"
  )
  find_library(SIMDJSON_LIBRARY simdjson
    HINTS
      "$ENV{CONDA_PREFIX}/lib"
      "$ENV{HOME}/miniconda3/lib"
      "$ENV{HOME}/miniconda3/envs/verl-cpu/lib"
  )
endif()
set(CVERL_HAS_SIMDJSON OFF)
if (simdjson_FOUND OR (SIMDJSON_INCLUDE_DIR AND SIMDJSON_LIBRARY))
  set(CVERL_HAS_SIMDJSON ON)
endif()

if (NOT CVERL_HAS_SIMDJSON)
  find_package(nlohmann_json QUIET)
endif()
if (NOT CVERL_HAS_SIMDJSON AND NOT nlohmann_json_FOUND)
  find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
    HINTS
      "$ENV{CONDA_PREFIX}/include"
      "$ENV{HOME}/miniconda3/include"
      "$ENV{HOME}/miniconda3/envs/verl-cpu/include"
  )
  if (NOT NLOHMANN_JSON_INCLUDE_DIR)
    message(FATAL_ERROR "simdjson or nlohmann/json.hpp is required for JSONL dataset loading.")
  endif()
endif()

if (CVERL_ENABLE_NCCL)
  find_package(Python COMPONENTS Interpreter REQUIRED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import site; print(';'.join(site.getsitepackages() + [site.getusersitepackages()]))"
    OUTPUT_VARIABLE CVERL_PYTHON_SITE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(CVERL_NCCL_HINT_DIRS)
  foreach(site_dir IN LISTS CVERL_PYTHON_SITE_DIRS)
    list(APPEND CVERL_NCCL_HINT_DIRS
      "${site_dir}/nvidia/nccl/include"
      "${site_dir}/nvidia/nccl/lib"
    )
  endforeach()
  find_path(NCCL_INCLUDE_DIR nccl.h HINTS ${CVERL_NCCL_HINT_DIRS} NO_DEFAULT_PATH)
  find_library(NCCL_LIBRARY NAMES nccl libnccl.so.2 HINTS ${CVERL_NCCL_HINT_DIRS} NO_DEFAULT_PATH)
  if (NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
    find_path(NCCL_INCLUDE_DIR nccl.h)
    find_library(NCCL_LIBRARY nccl)
  endif()
  if (NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
    message(FATAL_ERROR "CVERL_ENABLE_NCCL=ON requires nccl.h and libnccl")
  endif()
  message(STATUS "Found NCCL: ${NCCL_LIBRARY}")
endif()

add_library(cverl STATIC
  src/status.cc
  src/hf_downloader.cc
  src/data/hf_dataset.cc
  src/distributed/collectives.cc
  src/distributed/parallel_ops.cc
  src/distributed/topology.cc
  src/model/hf_model_loader.cc
  src/model/qwen3_5_text.cc
  src/core_algos_c_api.cc
  src/torch/core_algos_torch.cc
  src/torch/simple_grpo_trainer.cc
)
target_include_directories(cverl PUBLIC include)
if (CVERL_HAS_SIMDJSON AND NOT simdjson_FOUND)
  target_include_directories(cverl PUBLIC ${SIMDJSON_INCLUDE_DIR})
endif()
if (NOT CVERL_HAS_SIMDJSON AND NOT nlohmann_json_FOUND)
  target_include_directories(cverl PUBLIC ${NLOHMANN_JSON_INCLUDE_DIR})
endif()
target_link_libraries(cverl PUBLIC "${TORCH_LIBRARIES}")
if (CVERL_HAS_SIMDJSON)
  target_compile_definitions(cverl PUBLIC CVERL_HAS_SIMDJSON=1)
  if (simdjson_FOUND)
    target_link_libraries(cverl PUBLIC simdjson::simdjson)
  else()
    target_link_libraries(cverl PUBLIC ${SIMDJSON_LIBRARY})
  endif()
elseif (nlohmann_json_FOUND)
  target_link_libraries(cverl PUBLIC nlohmann_json::nlohmann_json)
endif()
target_compile_options(cverl PRIVATE ${TORCH_CXX_FLAGS})
target_compile_definitions(cverl PRIVATE
  CVERL_HF_DOWNLOAD_SCRIPT="${CMAKE_CURRENT_SOURCE_DIR}/tools/hf_snapshot_download.py"
  CVERL_HF_DATASET_SCRIPT="${CMAKE_CURRENT_SOURCE_DIR}/tools/hf_dataset_download.py"
)

if (CVERL_ENABLE_CUDA)
  enable_language(CUDA)
  target_sources(cverl PRIVATE
    cuda/core_algos_cuda.cu
  )
endif()

if (CVERL_ENABLE_NCCL)
  target_sources(cverl PRIVATE
    src/distributed/nccl_collectives.cc
  )
  target_include_directories(cverl PUBLIC ${NCCL_INCLUDE_DIR})
  target_link_libraries(cverl PUBLIC ${NCCL_LIBRARY})
  target_compile_definitions(cverl PUBLIC CVERL_ENABLE_NCCL=1)
endif()

if (CVERL_BUILD_TESTS)
  add_executable(test_core_algos_cpu tests/test_core_algos_cpu.cc)
  target_link_libraries(test_core_algos_cpu PRIVATE cverl)

  add_executable(compare_golden tools/compare_golden.cc)
  target_link_libraries(compare_golden PRIVATE cverl)

  add_executable(hf_download tools/hf_download.cc)
  target_link_libraries(hf_download PRIVATE cverl)

  add_executable(hf_dataset_download tools/data/hf_dataset_download.cc)
  target_link_libraries(hf_dataset_download PRIVATE cverl)

  add_executable(inspect_hf_model tools/model/inspect_hf_model.cc)
  target_link_libraries(inspect_hf_model PRIVATE cverl)

  add_executable(qwen3_5_forward tools/model/qwen3_5_forward.cc)
  target_link_libraries(qwen3_5_forward PRIVATE cverl)
  target_compile_options(qwen3_5_forward PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(test_torch_backend tests/torch/test_torch_backend.cc)
  target_link_libraries(test_torch_backend PRIVATE cverl)
  target_compile_options(test_torch_backend PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(test_simple_grpo_trainer tests/torch/test_simple_grpo_trainer.cc)
  target_link_libraries(test_simple_grpo_trainer PRIVATE cverl)
  target_compile_options(test_simple_grpo_trainer PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(test_hf_model_loader tests/model/test_hf_model_loader.cc)
  target_link_libraries(test_hf_model_loader PRIVATE cverl)
  target_compile_options(test_hf_model_loader PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(test_hf_dataset tests/data/test_hf_dataset.cc)
  target_link_libraries(test_hf_dataset PRIVATE cverl)

  add_executable(test_distributed_topology tests/distributed/test_topology.cc)
  target_link_libraries(test_distributed_topology PRIVATE cverl)
  target_compile_options(test_distributed_topology PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(test_parallel_ops tests/distributed/test_parallel_ops.cc)
  target_link_libraries(test_parallel_ops PRIVATE cverl)
  target_compile_options(test_parallel_ops PRIVATE ${TORCH_CXX_FLAGS})

  if (CVERL_ENABLE_NCCL)
    add_executable(test_nccl_collectives tests/distributed/test_nccl_collectives.cc)
    target_link_libraries(test_nccl_collectives PRIVATE cverl)
    target_compile_options(test_nccl_collectives PRIVATE ${TORCH_CXX_FLAGS})

    add_executable(qwen3_5_tp_smoke tools/model/qwen3_5_tp_smoke.cc)
    target_link_libraries(qwen3_5_tp_smoke PRIVATE cverl)
    target_compile_options(qwen3_5_tp_smoke PRIVATE ${TORCH_CXX_FLAGS})

    add_executable(qwen3_5_dp_tp_smoke tools/model/qwen3_5_dp_tp_smoke.cc)
    target_link_libraries(qwen3_5_dp_tp_smoke PRIVATE cverl)
    target_compile_options(qwen3_5_dp_tp_smoke PRIVATE ${TORCH_CXX_FLAGS})
  endif()

  add_executable(minimal_ppo_step examples/torch/minimal_ppo_step.cc)
  target_link_libraries(minimal_ppo_step PRIVATE cverl)
  target_compile_options(minimal_ppo_step PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(toy_grpo_trainer examples/torch/toy_grpo_trainer.cc)
  target_link_libraries(toy_grpo_trainer PRIVATE cverl)
  target_compile_options(toy_grpo_trainer PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(hf_embedding_grpo_trainer examples/model/hf_embedding_grpo_trainer.cc)
  target_link_libraries(hf_embedding_grpo_trainer PRIVATE cverl)
  target_compile_options(hf_embedding_grpo_trainer PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(cverl_gsm8k_grpo_bench examples/bench/cverl_gsm8k_grpo_bench.cc)
  target_link_libraries(cverl_gsm8k_grpo_bench PRIVATE cverl)
  target_compile_options(cverl_gsm8k_grpo_bench PRIVATE ${TORCH_CXX_FLAGS})
endif()
