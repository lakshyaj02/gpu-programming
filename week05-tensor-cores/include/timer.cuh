#pragma once

#include "common.cuh"

class GpuTimer {
public:
    GpuTimer();
    ~GpuTimer();

    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start(cudaStream_t stream = nullptr);
    float stop(cudaStream_t stream = nullptr);

private:
    cudaEvent_t startEvent_{};
    cudaEvent_t stopEvent_{};
};

inline GpuTimer::GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&startEvent_));
    CUDA_CHECK(cudaEventCreate(&stopEvent_));
}

inline GpuTimer::~GpuTimer() {
    cudaEventDestroy(startEvent_);
    cudaEventDestroy(stopEvent_);
}

inline void GpuTimer::start(cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(startEvent_, stream));
}

inline float GpuTimer::stop(cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(stopEvent_, stream));
    CUDA_CHECK(cudaEventSynchronize(stopEvent_));
    float elapsedMilliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, startEvent_, stopEvent_));
    return elapsedMilliseconds;
}