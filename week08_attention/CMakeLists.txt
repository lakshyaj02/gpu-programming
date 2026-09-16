cmake_minimum_required(VERSION 3.24)

project(
    week08_attention
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(
    week08_attention_kernels
    STATIC
    src/qk_matmul.cu
    src/scale_mask.cu
    src/softmax.cu
    src/pv_matmul.cu
    src/attention_naive.cu
    src/attention_tiled.cu
)
target_include_directories(
    week08_attention_kernels
    PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/include
)
set_target_properties(
    week08_attention_kernels
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)

add_executable(week08_attention_benchmark benchmark/benchmark.cu)
target_link_libraries(week08_attention_benchmark PRIVATE week08_attention_kernels)
set_target_properties(
    week08_attention_benchmark
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)