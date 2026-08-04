cmake_minimum_required(VERSION 3.24)

project(
    week01_vector_ops
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(
    week01_benchmark
    src/main.cu
    src/vector_add.cu
    src/saxpy.cu
)

target_include_directories(
    week01_benchmark
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../common
    ${CMAKE_CURRENT_LIST_DIR}/include
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