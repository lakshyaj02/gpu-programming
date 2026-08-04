cmake_minimum_required(VERSION 3.24)

project(
    gpu_programming
    LANGUAGES CXX CUDA
)

add_subdirectory(week01-vector-ops)
add_subdirectory(week02-memory-tiling)
add_subdirectory(week03-reductions)