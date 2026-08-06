#include <cuda_profiler_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cuda_check.h"
#include "gemm.cuh"

static void checkCublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error: %s failed with status %d\n", operation, static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "Usage: %s <naive|shared|thread_coarse|cublas> <M> <N> <K> <tile_size>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* kernelName = argv[1];
    const int M = std::atoi(argv[2]);
    const int N = std::atoi(argv[3]);
    const int K = std::atoi(argv[4]);
    const int tileSize = std::atoi(argv[5]);
    const bool isNaive = std::strcmp(kernelName, "naive") == 0;
    const bool isShared = std::strcmp(kernelName, "shared") == 0;
    const bool isThreadCoarse = std::strcmp(kernelName, "thread_coarse") == 0;
    const bool isCublas = std::strcmp(kernelName, "cublas") == 0;

    if ((!isNaive && !isShared && !isThreadCoarse && !isCublas) || M <= 0 || N <= 0 || K <= 0 ||
        (!isCublas && tileSize != 8 && tileSize != 16 && tileSize != 32)) {
        std::fprintf(stderr, "Invalid arguments: dimensions must be positive and tile size must be 8, 16, or 32\n");
        return EXIT_FAILURE;
    }

    const std::size_t bytesA = static_cast<std::size_t>(M) * K * sizeof(float);
    const std::size_t bytesB = static_cast<std::size_t>(K) * N * sizeof(float);
    const std::size_t bytesC = static_cast<std::size_t>(M) * N * sizeof(float);
    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceC = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceA, bytesA));
    CUDA_CHECK(cudaMalloc(&deviceB, bytesB));
    CUDA_CHECK(cudaMalloc(&deviceC, bytesC));
    CUDA_CHECK(cudaMemset(deviceA, 0, bytesA));
    CUDA_CHECK(cudaMemset(deviceB, 0, bytesB));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cublasHandle_t cublasHandle = nullptr;
    checkCublas(cublasCreate(&cublasHandle), "cublasCreate");
    checkCublas(cublasSetStream(cublasHandle, stream), "cublasSetStream");

    const dim3 blockDim(tileSize, tileSize);
    auto launchKernel = [&]() {
        if (isNaive) {
            launchNaiveGemmKernel(deviceA, deviceB, deviceC, M, N, K, blockDim, stream);
        } else if (isShared) {
            launchSharedGemmKernel(deviceA, deviceB, deviceC, M, N, K, blockDim, stream);
        } else if (isThreadCoarse) {
            launchThreadCoarseGemmKernel(deviceA, deviceB, deviceC, M, N, K, tileSize, stream);
        } else {
            launchCublasGemm(cublasHandle, deviceA, deviceB, deviceC, M, N, K);
        }
    };

    launchKernel();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaProfilerStart());
    launchKernel();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaProfilerStop());

    checkCublas(cublasDestroy(cublasHandle), "cublasDestroy");
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(deviceA));
    CUDA_CHECK(cudaFree(deviceB));
    CUDA_CHECK(cudaFree(deviceC));
    return EXIT_SUCCESS;
}