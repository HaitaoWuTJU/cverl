cmake_minimum_required(VERSION 3.18)
project(cverl LANGUAGES CXX)

option(CVERL_BUILD_TESTS "Build cverl tests" ON)
option(CVERL_ENABLE_CUDA "Build CUDA kernels" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(cverl STATIC
  src/status.cc
  src/core_algos_cpu.cc
)
target_include_directories(cverl PUBLIC include)

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
endif()
