#pragma once

#include <cuda_runtime.h>

void launch_bias_relu_unfused(const float* input, const float* bias, float* scratch,
                              float* output, int rows, int columns,
                              cudaStream_t stream = nullptr);
void launch_bias_relu_fused(const float* input, const float* bias, float* output,
                            int rows, int columns, cudaStream_t stream = nullptr);

void launch_bias_gelu_unfused(const float* input, const float* bias, float* scratch,
                              float* output, int rows, int columns,
                              cudaStream_t stream = nullptr);
void launch_bias_gelu_fused(const float* input, const float* bias, float* output,
                            int rows, int columns, cudaStream_t stream = nullptr);

void launch_rmsnorm_unfused(const float* input, const float* gamma, float* scratch,
                            float* row_stats, float* output, int rows, int columns,
                            cudaStream_t stream = nullptr);
void launch_rmsnorm_fused(const float* input, const float* gamma, float* output,
                          int rows, int columns, cudaStream_t stream = nullptr);

void launch_layernorm_unfused(const float* input, const float* gamma, const float* beta,
                              float* scratch, float* row_stats, float* output,
                              int rows, int columns, cudaStream_t stream = nullptr);
void launch_layernorm_fused(const float* input, const float* gamma, const float* beta,
                            float* output, int rows, int columns,
                            cudaStream_t stream = nullptr);

void launch_residual_rmsnorm_unfused(const float* input, const float* residual,
                                     const float* gamma, float* scratch, float* row_stats,
                                     float* output, int rows, int columns,
                                     cudaStream_t stream = nullptr);
void launch_residual_rmsnorm_fused(const float* input, const float* residual,
                                   const float* gamma, float* output, int rows, int columns,
                                   cudaStream_t stream = nullptr);

void launch_softmax_unfused(const float* input, float* scratch, float* row_stats,
                            float* output, int rows, int columns,
                            cudaStream_t stream = nullptr);
void launch_softmax_fused(const float* input, float* output, int rows, int columns,
                          cudaStream_t stream = nullptr);