cmake_minimum_required(VERSION 3.14)
project(FloatLLM)

set(CMAKE_CXX_STANDARD 17)

# Require the user to define where their ggml folder is located
if(NOT DEFINED GGML_DIR)
    message(FATAL_ERROR "\n❌ GGML_DIR is not defined!\n FloatLLM requires the ggml library to compile. \nPlease download the ggml and pass its path to CMake using: -DGGML_DIR=/path/to/ggml\n")
endif()

add_subdirectory(${GGML_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ggml_build)

# build c++ bridge as a shared library
add_library(floatllm_compute SHARED floatllm_compute.cpp)

set_target_properties(floatllm_compute PROPERTIES PREFIX "")

set_target_properties(floatllm_compute PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(floatllm_compute PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

target_include_directories(floatllm_compute PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# Base linkages
target_link_libraries(floatllm_compute PRIVATE ggml ggml-base ggml-cpu)

# Hardware override and extensions
if(APPLE)
    target_link_libraries(floatllm_compute PRIVATE ggml-metal)
endif()

if(GGML_CUDA)
    target_link_libraries(floatllm_compute PRIVATE ggml-cuda)
endif()

if(GGML_VULKAN)
    target_link_libraries(floatllm_compute PRIVATE ggml-vulkan)
endif()

if(GGML_OPENCL)
    target_link_libraries(floatllm_compute PRIVATE ggml-opencl)
endif()

if(GGML_HIP)
    target_link_libraries(floatllm_compute PRIVATE ggml-hip)
endif()

if(GGML_SYCL)
    target_link_libraries(floatllm_compute PRIVATE ggml-sycl)
endif()

if(GGML_KOMPUTE)
    target_link_libraries(floatllm_compute PRIVATE ggml-kompute)
endif()