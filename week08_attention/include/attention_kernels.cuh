#pragma once

#include <cuda_runtime.h>

__global__ void qk_matmul_kernel(const float* q, const float* k, float* scores,
                                 int m, int n, int d);
__global__ void scale_mask_kernel(float* scores, int m, int n, float scale,
                                  bool causal);
__global__ void softmax_kernel(const float* scores, float* probabilities,
                               int m, int n);
__global__ void pv_matmul_kernel(const float* probabilities, const float* v,
                                 float* output, int m, int n, int dv);

void launch_qk_matmul(const float* q, const float* k, float* scores,
                      int m, int n, int d, cudaStream_t stream = nullptr);
void launch_scale_mask(float* scores, int m, int n, float scale, bool causal,
                       cudaStream_t stream = nullptr);
void launch_softmax(const float* scores, float* probabilities, int m, int n,
                    cudaStream_t stream = nullptr);
void launch_pv_matmul(const float* probabilities, const float* v, float* output,
                      int m, int n, int dv, cudaStream_t stream = nullptr);

void launch_attention_naive(const float* q, const float* k, const float* v,
                            float* scores, float* probabilities, float* output,
                            int m, int n, int d, int dv, float scale,
                            bool causal, cudaStream_t stream = nullptr);
void launch_attention_tiled(const float* q, const float* k, const float* v,
                            float* output, int m, int n, int d, int dv,
                            float scale, bool causal,
                            cudaStream_t stream = nullptr);