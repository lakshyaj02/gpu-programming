cmake_minimum_required(VERSION 3.24)

project(
    week04_gemm
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(
    week04_benchmark
    src/main.cu
    src/naive_multiply.cu
    src/shared_multiply.cu
    src/thread_coarse_multiply.cu
    src/cublas_multiply.cu
)

add_executable(
    week04_profile
    src/profile.cu
    src/naive_multiply.cu
    src/shared_multiply.cu
    src/thread_coarse_multiply.cu
    src/cublas_multiply.cu
)

foreach(target week04_benchmark week04_profile)
    target_include_directories(
        ${target}
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/../common
        ${CMAKE_CURRENT_LIST_DIR}/include
    )

    target_link_libraries(
        ${target}
        PRIVATE
        CUDA::cublas
    )

    target_compile_options(
        ${target}
        PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )

    set_target_properties(
        ${target}
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endforeach()