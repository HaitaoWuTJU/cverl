cmake_minimum_required(VERSION 3.18)
project(cverl LANGUAGES CXX)

option(CVERL_BUILD_TESTS "Build cverl tests" ON)
option(CVERL_ENABLE_CUDA "Build CUDA kernels" OFF)
option(CVERL_ENABLE_TORCH "Build optional LibTorch backend" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(cverl STATIC
  src/status.cc
  src/core_algos_cpu.cc
)
target_include_directories(cverl PUBLIC include)

if (CVERL_ENABLE_TORCH)
  find_package(Torch REQUIRED)
  add_library(cverl_torch STATIC
    src/torch/core_algos_torch.cc
  )
  target_include_directories(cverl_torch PUBLIC include)
  target_link_libraries(cverl_torch PUBLIC cverl "${TORCH_LIBRARIES}")
  target_compile_options(cverl_torch PRIVATE ${TORCH_CXX_FLAGS})
endif()

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

  if (CVERL_ENABLE_TORCH)
    add_executable(test_torch_backend tests/torch/test_torch_backend.cc)
    target_link_libraries(test_torch_backend PRIVATE cverl_torch)
    target_compile_options(test_torch_backend PRIVATE ${TORCH_CXX_FLAGS})

    add_executable(minimal_ppo_step examples/torch/minimal_ppo_step.cc)
    target_link_libraries(minimal_ppo_step PRIVATE cverl_torch)
    target_compile_options(minimal_ppo_step PRIVATE ${TORCH_CXX_FLAGS})
  endif()
endif()
