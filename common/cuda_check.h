#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t error_ = (call);                                     \
        if (error_ != cudaSuccess) {                                     \
            std::cerr << "CUDA error at " << __FILE__ << ":"             \
                      << __LINE__ << ": "                                 \
                      << cudaGetErrorString(error_) << '\n';              \
            std::exit(EXIT_FAILURE);                                     \
        }                                                                \
    } while (false)

#define CUDA_CHECK_KERNEL() do {                 \
    CUDA_CHECK(cudaGetLastError());              \
    CUDA_CHECK(cudaDeviceSynchronize());         \
} while (false)
