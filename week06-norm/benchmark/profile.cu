#include "norm_kernels.cuh"

#include <cuda_profiler_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "Usage: %s <naive|warp|warp_half> <hidden_size> <block_size>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char* kernel = argv[1];
    const int hiddenSize = std::atoi(argv[2]);
    const int blockSize = std::atoi(argv[3]);
    const bool isNaive = std::strcmp(kernel, "naive") == 0;
    const bool isWarp = std::strcmp(kernel, "warp") == 0;
    const bool isHalf = std::strcmp(kernel, "warp_half") == 0;
    if ((!isNaive && !isWarp && !isHalf) || hiddenSize <= 0 || blockSize < 32 || blockSize > 1024 || blockSize % 32 != 0) {
        std::fprintf(stderr, "Invalid kernel, hidden size, or block size\n");
        return EXIT_FAILURE;
    }

    const std::size_t bytes = static_cast<std::size_t>(hiddenSize) * (isHalf ? sizeof(half) : sizeof(float));
    void* input = nullptr;
    void* output = nullptr;
    void* gamma = nullptr;
    void* beta = nullptr;
    cudaMalloc(&input, bytes); cudaMalloc(&output, bytes); cudaMalloc(&gamma, bytes); cudaMalloc(&beta, bytes);
    cudaMemset(input, 0, bytes); cudaMemset(gamma, 0, bytes); cudaMemset(beta, 0, bytes);
    auto launch = [&] {
        if (isNaive) launch_naive_rmsnorm(static_cast<float*>(input), static_cast<float*>(output), static_cast<float*>(gamma), static_cast<float*>(beta), hiddenSize, blockSize);
        else if (isWarp) launch_warp_rmsnorm(static_cast<float*>(input), static_cast<float*>(output), static_cast<float*>(gamma), static_cast<float*>(beta), hiddenSize, blockSize);
        else launch_warp_rmsnorm_half(static_cast<half*>(input), static_cast<half*>(output), static_cast<half*>(gamma), static_cast<half*>(beta), hiddenSize, blockSize);
    };
    launch();
    cudaProfilerStart();
    launch();
    cudaProfilerStop();
    cudaFree(input); cudaFree(output); cudaFree(gamma); cudaFree(beta);
    return EXIT_SUCCESS;
}