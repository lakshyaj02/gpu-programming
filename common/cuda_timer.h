#pragma once

#include <cuda_runtime.h>

#include "cuda_check.h"

class CudaTimer {
public:
	CudaTimer() {
		CUDA_CHECK(cudaEventCreate(&startEvent_));
		CUDA_CHECK(cudaEventCreate(&stopEvent_));
	}

	~CudaTimer() {
		cudaEventDestroy(startEvent_);
		cudaEventDestroy(stopEvent_);
	}

	CudaTimer(const CudaTimer&) = delete;
	CudaTimer& operator=(const CudaTimer&) = delete;

	void start(cudaStream_t stream = nullptr) {
		CUDA_CHECK(cudaEventRecord(startEvent_, stream));
	}

	float stop(cudaStream_t stream = nullptr) {
		CUDA_CHECK(cudaEventRecord(stopEvent_, stream));
		CUDA_CHECK(cudaEventSynchronize(stopEvent_));

		float elapsedMs = 0.0f;
		CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, startEvent_, stopEvent_));
		return elapsedMs;
	}

private:
	cudaEvent_t startEvent_{};
	cudaEvent_t stopEvent_{};
};
