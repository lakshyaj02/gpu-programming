#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

inline void checkCuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s failed: %s\n",
                     file, line, expression, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

inline void checkCublas(cublasStatus_t status, const char* expression,
                        const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error at %s:%d: %s failed with status %d\n",
                     file, line, expression, static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression, __FILE__, __LINE__)
#define CUBLAS_CHECK(expression) checkCublas((expression), #expression, __FILE__, __LINE__)

void runPrecisionDemo();

void launchCublasGemm(cublasHandle_t handle, const half* matrixA, const half* matrixB,
                      float* matrixC, int m, int n, int k);

void launchWmmaGemm(const half* matrixA, const half* matrixB, float* matrixC,
                    int m, int n, int k, cudaStream_t stream);

void launchWmmaTf32Gemm(const float* matrixA, const float* matrixB, float* matrixC,
                        int m, int n, int k, cudaStream_t stream);

void launchWmmaBf16Gemm(const __nv_bfloat16* matrixA, const __nv_bfloat16* matrixB,
                        float* matrixC, int m, int n, int k, cudaStream_t stream);