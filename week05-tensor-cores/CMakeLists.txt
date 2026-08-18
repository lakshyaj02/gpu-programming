cmake_minimum_required(VERSION 3.24)

project(
    week05_tensor_cores
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(
    week05_tensor_cores
    src/main.cu
    src/precision_demo.cu
    src/cublas_gemm.cu
    src/wmma_gemm.cu
)

target_include_directories(
    week05_tensor_cores
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_libraries(
    week05_tensor_cores
    PRIVATE
    CUDA::cublas
)

target_compile_options(
    week05_tensor_cores
    PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

set_target_properties(
    week05_tensor_cores
    PROPERTIES
    CUDA_ARCHITECTURES "100;120"
    CUDA_SEPARABLE_COMPILATION ON
)