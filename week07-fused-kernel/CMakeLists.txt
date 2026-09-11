cmake_minimum_required(VERSION 3.24)

project(
    week07_fused_kernel
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(
    week07_kernels
    STATIC
    src/bias_relu.cu
    src/bias_gelu.cu
    src/rmsnorm.cu
    src/layernorm.cu
    src/residual_rmsnorm.cu
    src/softmax.cu
)
target_include_directories(week07_kernels PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_compile_options(week07_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
set_target_properties(week07_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(week07_fused_test src/main.cu)
target_link_libraries(week07_fused_test PRIVATE week07_kernels)

add_executable(week07_benchmark benchmark/benchmark.cu)
target_link_libraries(week07_benchmark PRIVATE week07_kernels)

target_compile_options(
    week07_fused_test
    PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
set_target_properties(week07_fused_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(
    week07_benchmark
    PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
set_target_properties(week07_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)