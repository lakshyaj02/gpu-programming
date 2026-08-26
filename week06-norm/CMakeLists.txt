cmake_minimum_required(VERSION 3.24)

project(
    week06_norm
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(
    week06_benchmark
    benchmark/benchmark.cu
    src/rmsnorm_naive.cu
    src/rmsnorm_warp.cu
    src/rmsnorm_half.cu
    src/rmsnorm_vectorized.cu
    src/layernorm.cu
    src/fused_residual_rmsnorm.cu
)

add_executable(
    week06_profile
    benchmark/profile.cu
    src/rmsnorm_naive.cu
    src/rmsnorm_warp.cu
    src/rmsnorm_half.cu
    src/rmsnorm_vectorized.cu
    src/layernorm.cu
    src/fused_residual_rmsnorm.cu
)

add_executable(
    week06_norm_test
    src/main.cu
    src/rmsnorm_naive.cu
    src/rmsnorm_warp.cu
    src/rmsnorm_half.cu
    src/rmsnorm_vectorized.cu
    src/layernorm.cu
    src/fused_residual_rmsnorm.cu
)

foreach(target week06_benchmark week06_profile week06_norm_test)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)
    target_compile_options(
        ${target}
        PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
    set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()