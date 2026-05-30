cmake_minimum_required(VERSION 3.18)
project(cverl LANGUAGES CXX)

option(CVERL_BUILD_TESTS "Build cverl tests" ON)
option(CVERL_ENABLE_CUDA "Build CUDA kernels" OFF)
option(CVERL_ENABLE_NCCL "Build NCCL collectives" OFF)
option(CVERL_BUILD_PYTHON_BRIDGE "Build Python bridge for vLLM Native RL weight sync" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)
find_package(Threads REQUIRED)
if (CVERL_BUILD_PYTHON_BRIDGE)
  find_package(Python COMPONENTS Interpreter QUIET)
  if (Python_Interpreter_FOUND)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -c
              "import sysconfig; print(sysconfig.get_paths()['include']); print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
      OUTPUT_VARIABLE CVERL_PYTHON_CONFIG
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REPLACE "\n" ";" CVERL_PYTHON_CONFIG_LIST "${CVERL_PYTHON_CONFIG}")
    list(GET CVERL_PYTHON_CONFIG_LIST 0 CVERL_PYTHON_INCLUDE_DIR)
    list(GET CVERL_PYTHON_CONFIG_LIST 1 CVERL_PYTHON_EXT_SUFFIX)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -c
              "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')"
      OUTPUT_VARIABLE CVERL_TORCH_PYTHON_LIB_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -c
              "import torch.utils.cpp_extension as e; print(';'.join(e.include_paths()))"
      OUTPUT_VARIABLE CVERL_TORCH_EXTENSION_INCLUDES
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    find_library(TORCH_PYTHON_LIBRARY torch_python HINTS "${CVERL_TORCH_PYTHON_LIB_DIR}" NO_DEFAULT_PATH)
  endif()
  if (NOT Python_Interpreter_FOUND OR NOT CVERL_PYTHON_INCLUDE_DIR OR NOT TORCH_PYTHON_LIBRARY)
    message(WARNING "Python bridge disabled: requires Python headers and libtorch_python")
    set(CVERL_BUILD_PYTHON_BRIDGE OFF)
  endif()
endif()

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

# nlohmann/json is required by the HF tokenizer and dataset loader. simdjson is
# preferred for the JSONL dataset loader hot path when available.
find_package(nlohmann_json QUIET)
if (NOT nlohmann_json_FOUND)
  find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
    HINTS
      "$ENV{CONDA_PREFIX}/include"
      "$ENV{HOME}/miniconda3/include"
      "$ENV{HOME}/miniconda3/envs/verl-cpu/include"
  )
  if (NOT NLOHMANN_JSON_INCLUDE_DIR)
    message(FATAL_ERROR "nlohmann/json.hpp is required (header at ${NLOHMANN_JSON_INCLUDE_DIR}).")
  endif()
endif()

# PCRE2 powers the byte-level BPE pre-tokenizer regex (Unicode property
# classes \p{L}/\p{N} required by GPT-2 / Qwen3 patterns; std::regex does
# not support them and is also linear-time slow on these inputs).
find_path(PCRE2_INCLUDE_DIR pcre2.h
  HINTS
    "$ENV{CONDA_PREFIX}/include"
    "$ENV{HOME}/miniconda3/include"
    /usr/include
)
find_library(PCRE2_LIBRARY NAMES pcre2-8 libpcre2-8
  HINTS
    "$ENV{CONDA_PREFIX}/lib"
    "$ENV{HOME}/miniconda3/lib"
    /usr/lib64 /usr/lib
)
if (NOT PCRE2_INCLUDE_DIR OR NOT PCRE2_LIBRARY)
  message(FATAL_ERROR "PCRE2 (libpcre2-8) is required for HfBpeTokenizer")
endif()
message(STATUS "Found PCRE2: ${PCRE2_LIBRARY}")

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
  src/distributed/weight_sync.cc
  src/model/hf_model_loader.cc
  src/model/qwen3_5_text.cc
  src/model/safetensors_writer.cc
  src/reward/gsm8k.cc
  src/rollout/rollout_batch.cc
  src/rollout/policy_worker.cc
  src/rollout/plugin_worker.cc
  src/rollout/shared_memory.cc
  src/rollout/shared_memory_transport.cc
  src/rollout/transport.cc
  src/rollout/worker.cc
  src/text/byte_tokenizer.cc
  src/text/hf_bpe_tokenizer.cc
  src/core_algos_c_api.cc
  src/torch/causal_lm_policy.cc
  src/torch/core_algos_torch.cc
  src/torch/qwen3_5_causal_lm_policy.cc
  src/torch/simple_grpo_trainer.cc
  src/torch/tiny_causal_lm_policy.cc
  src/torch/tiny_causal_policy.cc
)
set_target_properties(cverl PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(cverl PUBLIC include)
if (CVERL_HAS_SIMDJSON AND NOT simdjson_FOUND)
  target_include_directories(cverl PUBLIC ${SIMDJSON_INCLUDE_DIR})
endif()
if (NOT nlohmann_json_FOUND)
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
endif()
if (nlohmann_json_FOUND)
  target_link_libraries(cverl PUBLIC nlohmann_json::nlohmann_json)
endif()
target_link_libraries(cverl PUBLIC Threads::Threads)
# librt provides POSIX shared memory + named semaphores on Linux.
target_link_libraries(cverl PUBLIC rt)
target_link_libraries(cverl PUBLIC ${CMAKE_DL_LIBS})
target_include_directories(cverl PUBLIC ${PCRE2_INCLUDE_DIR})
target_link_libraries(cverl PUBLIC ${PCRE2_LIBRARY})
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

if (CVERL_BUILD_PYTHON_BRIDGE)
  add_library(_cverl_vllm_bridge MODULE python/cverl_vllm_bridge.cc)
  target_include_directories(_cverl_vllm_bridge PRIVATE
    ${CVERL_PYTHON_INCLUDE_DIR}
    ${CVERL_TORCH_EXTENSION_INCLUDES}
  )
  target_link_libraries(_cverl_vllm_bridge PRIVATE cverl ${TORCH_PYTHON_LIBRARY})
  target_compile_options(_cverl_vllm_bridge PRIVATE ${TORCH_CXX_FLAGS})
  set_target_properties(_cverl_vllm_bridge PROPERTIES
    PREFIX ""
    SUFFIX "${CVERL_PYTHON_EXT_SUFFIX}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
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

  add_executable(test_safetensors_writer tests/model/test_safetensors_writer.cc)
  target_link_libraries(test_safetensors_writer PRIVATE cverl)
  target_compile_options(test_safetensors_writer PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(test_hf_dataset tests/data/test_hf_dataset.cc)
  target_link_libraries(test_hf_dataset PRIVATE cverl)

  add_executable(test_rollout_shared_memory tests/rollout/test_shared_memory.cc)
  target_link_libraries(test_rollout_shared_memory PRIVATE cverl)

  add_executable(test_rollout_shared_memory_transport tests/rollout/test_shared_memory_transport.cc)
  target_link_libraries(test_rollout_shared_memory_transport PRIVATE cverl)

  add_executable(test_rollout_transport tests/rollout/test_transport.cc)
  target_link_libraries(test_rollout_transport PRIVATE cverl)

  add_executable(test_rollout_worker tests/rollout/test_rollout_worker.cc)
  target_link_libraries(test_rollout_worker PRIVATE cverl)
  target_compile_options(test_rollout_worker PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(test_policy_rollout_worker tests/rollout/test_policy_rollout_worker.cc)
  target_link_libraries(test_policy_rollout_worker PRIVATE cverl)
  target_compile_options(test_policy_rollout_worker PRIVATE ${TORCH_CXX_FLAGS})

  add_library(test_rollout_worker_plugin_fixture SHARED tests/rollout/plugin_worker_fixture.cc)
  target_link_libraries(test_rollout_worker_plugin_fixture PRIVATE cverl)
  target_compile_options(test_rollout_worker_plugin_fixture PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(test_plugin_worker tests/rollout/test_plugin_worker.cc)
  target_link_libraries(test_plugin_worker PRIVATE cverl)
  target_compile_options(test_plugin_worker PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(test_gsm8k_reward tests/reward/test_gsm8k_reward.cc)
  target_link_libraries(test_gsm8k_reward PRIVATE cverl)

  add_executable(test_byte_tokenizer tests/text/test_byte_tokenizer.cc)
  target_link_libraries(test_byte_tokenizer PRIVATE cverl)

  add_executable(test_hf_bpe_tokenizer tests/text/test_hf_bpe_tokenizer.cc)
  target_link_libraries(test_hf_bpe_tokenizer PRIVATE cverl)
  target_compile_definitions(test_hf_bpe_tokenizer PRIVATE
    CVERL_PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

  add_executable(test_rollout_batch tests/rollout/test_rollout_batch.cc)
  target_link_libraries(test_rollout_batch PRIVATE cverl)
  target_compile_options(test_rollout_batch PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(test_gsm8k_grpo_trainer tests/rollout/test_gsm8k_grpo_trainer.cc)
  target_link_libraries(test_gsm8k_grpo_trainer PRIVATE cverl)
  target_compile_options(test_gsm8k_grpo_trainer PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(test_reference_policy_kl tests/torch/test_reference_policy_kl.cc)
  target_link_libraries(test_reference_policy_kl PRIVATE cverl)
  target_compile_options(test_reference_policy_kl PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(test_qwen3_5_causal_lm_policy tests/torch/test_qwen3_5_causal_lm_policy.cc)
  target_link_libraries(test_qwen3_5_causal_lm_policy PRIVATE cverl)
  target_compile_options(test_qwen3_5_causal_lm_policy PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(test_qwen3_5_grpo_step tests/torch/test_qwen3_5_grpo_step.cc)
  target_link_libraries(test_qwen3_5_grpo_step PRIVATE cverl)
  target_compile_options(test_qwen3_5_grpo_step PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(test_distributed_topology tests/distributed/test_topology.cc)
  target_link_libraries(test_distributed_topology PRIVATE cverl)
  target_compile_options(test_distributed_topology PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(test_parallel_ops tests/distributed/test_parallel_ops.cc)
  target_link_libraries(test_parallel_ops PRIVATE cverl)
  target_compile_options(test_parallel_ops PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(test_weight_sync tests/distributed/test_weight_sync.cc)
  target_link_libraries(test_weight_sync PRIVATE cverl)
  target_compile_options(test_weight_sync PRIVATE ${TORCH_CXX_FLAGS})

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

  add_executable(minimal_ppo_step tools/torch/minimal_ppo_step.cc)
  target_link_libraries(minimal_ppo_step PRIVATE cverl)
  target_compile_options(minimal_ppo_step PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(toy_grpo_trainer tools/torch/toy_grpo_trainer.cc)
  target_link_libraries(toy_grpo_trainer PRIVATE cverl)
  target_compile_options(toy_grpo_trainer PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(hf_embedding_grpo_trainer tools/model/hf_embedding_grpo_trainer.cc)
  target_link_libraries(hf_embedding_grpo_trainer PRIVATE cverl)
  target_compile_options(hf_embedding_grpo_trainer PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(cverl_gsm8k_grpo_bench tools/bench/cverl_gsm8k_grpo_bench.cc)
  target_link_libraries(cverl_gsm8k_grpo_bench PRIVATE cverl)
  target_compile_options(cverl_gsm8k_grpo_bench PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(vllm_weight_manifest tools/rollout/vllm_weight_manifest.cc)
  target_link_libraries(vllm_weight_manifest PRIVATE cverl)
  target_compile_options(vllm_weight_manifest PRIVATE ${TORCH_CXX_FLAGS})

  add_executable(gsm8k_grpo_trainer tools/rollout/gsm8k_grpo_trainer.cc)
  target_link_libraries(gsm8k_grpo_trainer PRIVATE cverl)
  target_compile_options(gsm8k_grpo_trainer PRIVATE ${TORCH_CXX_FLAGS})

endif()
