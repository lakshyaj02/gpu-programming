cmake_minimum_required(VERSION 3.24)

project(
    gpu_programming
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(
    week01_benchmark
    week01-vector-ops/src/main.cu
    week01-vector-ops/src/vector_add.cu
    week01-vector-ops/src/saxpy.cu
)

target_include_directories(
    week01_benchmark
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/week01-vector-ops/include
)

target_compile_options(
    week01_benchmark
    PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

set_target_properties(
    week01_benchmark
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(
    week02_benchmark
    week02-memory-tiling/src/main.cu
    week02-memory-tiling/src/memory_access.cu
    week02-memory-tiling/src/transpose.cu
    week02-memory-tiling/src/matmul.cu
)

target_include_directories(
    week02_benchmark
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/week02-memory-tiling/include
)

target_compile_options(
    week02_benchmark
    PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

set_target_properties(
    week02_benchmark
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)