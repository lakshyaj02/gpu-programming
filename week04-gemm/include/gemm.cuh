#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>

void launchNaiveGemmKernel(const float* A, const float* B, float* C, int M, int N, int K, dim3 blockDim, cudaStream_t stream);
void launchSharedGemmKernel(const float *A, const float *B, float *C, int M, int N, int K, dim3 blockDim, cudaStream_t stream);
void launchThreadCoarseGemmKernel(const float* A, const float* B, float* C, int M, int N, int K, int tileSize, cudaStream_t stream);
void launchCublasGemm(cublasHandle_t handle, const float* A, const float* B, float* C, int M, int N, int K);