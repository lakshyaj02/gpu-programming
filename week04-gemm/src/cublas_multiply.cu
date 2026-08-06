#include <cstdio>
#include <cstdlib>

#include "gemm.cuh"

void launchCublasGemm(cublasHandle_t handle,
                      const float* A,
                      const float* B,
                      float* C,
                      int M,
                      int N,
                      int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    const cublasStatus_t status = cublasSgemm(handle,
                                              CUBLAS_OP_N,
                                              CUBLAS_OP_N,
                                              N,
                                              M,
                                              K,
                                              &alpha,
                                              B,
                                              N,
                                              A,
                                              K,
                                              &beta,
                                              C,
                                              N);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error: cublasSgemm failed with status %d\n", static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}