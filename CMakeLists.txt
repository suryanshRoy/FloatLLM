cmake_minimum_required(VERSION 3.14)
project(FloatLLM)

include(FetchContent)
set(GGML_GIT_TAG "master" CACHE STRING "The Git branch, tag, or commit hash for GGML")

FetchContent_Declare(
    ggml
    GIT_REPOSITORY https://github.com/ggerganov/ggml.git
    GIT_TAG        ${GGML_GIT_TAG}
)

message(STATUS "[FloatLLM] Fetching GGML version: ${GGML_GIT_TAG}...")
FetchContent_MakeAvailable(ggml)

set(CMAKE_CXX_STANDARD 17)

set(FLOATLLM_CORE
    src/FloatEngine.cpp
    src/FloatTokenizer.cpp
    src/FloatLoader.cpp
    src/FloatUI.cpp
)

# shared library
add_library(FloatLLM SHARED ${FLOATLLM_CORE})

set_target_properties(FloatLLM PROPERTIES PREFIX "")

set_target_properties(FloatLLM PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(FloatLLM PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

target_include_directories(FloatLLM PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(FloatLLM PUBLIC ggml ggml-base ggml-cpu)

if(APPLE)
    target_link_libraries(FloatLLM PUBLIC ggml-metal)
endif()

if(GGML_CUDA)
    target_link_libraries(FloatLLM PUBLIC ggml-cuda)
endif()

if(GGML_VULKAN)
    target_link_libraries(FloatLLM PUBLIC ggml-vulkan)
endif()

if(GGML_OPENCL)
    target_link_libraries(FloatLLM PUBLIC ggml-opencl)
endif()

if(GGML_HIP)
    target_link_libraries(FloatLLM PUBLIC ggml-hip)
endif()

if(GGML_SYCL)
    target_link_libraries(FloatLLM PUBLIC ggml-sycl)
endif()

if(GGML_KOMPUTE)
    target_link_libraries(FloatLLM PUBLIC ggml-kompute)
endif()

add_executable(runFloatLLM src/FloatLLM.cpp)

set_target_properties(runFloatLLM PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(runFloatLLM PRIVATE FloatLLM)