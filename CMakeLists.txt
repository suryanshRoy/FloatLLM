cmake_minimum_required(VERSION 3.14)
project(FloatLLM)

set(CMAKE_CXX_STANDARD 17)
add_subdirectory(ggml)
# build c++ bridge as a shared library
add_library(floatllm_compute SHARED floatllm_compute.cpp)

set_target_properties(floatllm_compute PROPERTIES PREFIX "")

set_target_properties(floatllm_compute PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(floatllm_compute PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

target_include_directories(floatllm_compute PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/ggml/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src"
)

target_link_libraries(floatllm_compute PRIVATE ggml ggml-base ggml-cpu)

if(APPLE)
    target_link_libraries(floatllm_compute PRIVATE ggml-metal)
endif()

if(GGML_CUDA)
    target_link_libraries(floatllm_compute PRIVATE ggml-cuda)
endif()

if(GGML_VULKAN)
    target_link_libraries(floatllm_compute PRIVATE ggml-vulkan)
endif()
