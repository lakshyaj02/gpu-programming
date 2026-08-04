cmake_minimum_required(VERSION 3.24)

project(
    week03_reductions
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(
    week03_benchmark
    src/main.cu
    src/reduction_atomic.cu
    src/reduction_shared.cu
    src/reduction_warp.cu
    src/reduction_hierarchical.cu
)

target_include_directories(
    week03_benchmark
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../common
    ${CMAKE_CURRENT_LIST_DIR}/include
)

target_compile_options(
    week03_benchmark
    PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

set_target_properties(
    week03_benchmark
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)