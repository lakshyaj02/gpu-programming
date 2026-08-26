#include "norm_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

std::vector<float> referenceRmsnorm(const std::vector<float>& input,
                                    const std::vector<float>& gamma,
                                    const std::vector<float>& beta) {
    double squareSum = 0.0;
    for (float value : input) squareSum += static_cast<double>(value) * value;
    const float inverseRms = 1.0f / std::sqrt(static_cast<float>(squareSum / input.size()) + 1.0e-6f);
    std::vector<float> output(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        output[index] = input[index] * inverseRms * gamma[index] + beta[index];
    }
    return output;
}

std::vector<float> referenceLayernorm(const std::vector<float>& input) {
    double sum = 0.0;
    for (float value : input) sum += value;
    const double mean = sum / input.size();
    double squaredDifferenceSum = 0.0;
    for (float value : input) {
        const double difference = value - mean;
        squaredDifferenceSum += difference * difference;
    }
    const double inverseStandardDeviation =
        1.0 / std::sqrt(squaredDifferenceSum / input.size() + 1.0e-5);
    std::vector<float> output(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        output[index] = static_cast<float>((input[index] - mean) * inverseStandardDeviation);
    }
    return output;
}

std::vector<float> referenceFusedResidualRmsnorm(const std::vector<float>& input,
                                                 const std::vector<float>& residual,
                                                 const std::vector<float>& gamma) {
    double squareSum = 0.0;
    for (std::size_t index = 0; index < input.size(); ++index) {
        const double value = static_cast<double>(input[index]) + residual[index];
        squareSum += value * value;
    }
    const double inverseRms = 1.0 / std::sqrt(squareSum / input.size() + 1.0e-6);
    std::vector<float> output(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        output[index] = static_cast<float>((input[index] + residual[index]) *
                                           inverseRms * gamma[index]);
    }
    return output;
}

struct ErrorMetrics {
    float maxAbs = 0.0f;
    float maxRel = 0.0f;
    double rmse = 0.0;
    std::size_t errorCount = 0;
};

ErrorMetrics measureError(const std::vector<float>& actual, const std::vector<float>& expected,
                          float absoluteTolerance, float relativeTolerance) {
    ErrorMetrics metrics;
    double squaredError = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float absolute = std::abs(actual[index] - expected[index]);
        const float relative = absolute / (std::abs(expected[index]) + 1.0e-6f);
        metrics.maxAbs = std::max(metrics.maxAbs, absolute);
        metrics.maxRel = std::max(metrics.maxRel, relative);
        squaredError += static_cast<double>(absolute) * absolute;
        if (absolute > absoluteTolerance + relativeTolerance * std::abs(expected[index])) ++metrics.errorCount;
    }
    metrics.rmse = std::sqrt(squaredError / actual.size());
    return metrics;
}

template <typename Launch>
std::pair<float, float> timeKernel(int iterations, int warmupIterations, Launch launch) {
    for (int warmup = 0; warmup < warmupIterations; ++warmup) launch();
    checkCuda(cudaDeviceSynchronize(), "warmup synchronization");
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");
    float totalMs = 0.0f;
    float minMs = INFINITY;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        checkCuda(cudaEventRecord(start), "record start event");
        launch();
        checkCuda(cudaEventRecord(stop), "record stop event");
        checkCuda(cudaEventSynchronize(stop), "synchronize stop event");
        float elapsedMs = 0.0f;
        checkCuda(cudaEventElapsedTime(&elapsedMs, start, stop), "measure elapsed time");
        totalMs += elapsedMs;
        minMs = std::min(minMs, elapsedMs);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return {totalMs / iterations, minMs};
}

}  // namespace

int main(int argc, char** argv) {
    int iterations = 20;
    int warmupIterations = 5;
    if (argc > 1) iterations = std::atoi(argv[1]);
    if (argc > 2) warmupIterations = std::atoi(argv[2]);
    if (iterations <= 0 || warmupIterations < 0) {
        std::fprintf(stderr, "Usage: %s [iterations] [warmup_iterations]\n", argv[0]);
        return EXIT_FAILURE;
    }

    constexpr int blockSize = 256;
    constexpr float floatAtol = 1.0e-4f;
    constexpr float floatRtol = 1.0e-4f;
    constexpr float halfAtol = 5.0e-3f;
    constexpr float halfRtol = 5.0e-3f;
    int failedRuns = 0;
    std::puts("kernel,precision,hidden_size,block_size,iterations,avg_kernel_ms,min_kernel_ms,bandwidth_gbps,max_abs_error,max_rel_error,rmse,error_count,status");

    for (int hiddenSize : {768, 1024, 2048, 4096, 8192}) {
        std::mt19937 generator(42 + hiddenSize);
        std::uniform_real_distribution<float> inputDistribution(-2.0f, 2.0f);
        std::uniform_real_distribution<float> gammaDistribution(0.5f, 1.5f);
        std::uniform_real_distribution<float> betaDistribution(-0.25f, 0.25f);
        std::vector<float> input(hiddenSize), residual(hiddenSize), gamma(hiddenSize), beta(hiddenSize);
        for (int index = 0; index < hiddenSize; ++index) {
            input[index] = inputDistribution(generator);
            residual[index] = inputDistribution(generator);
            gamma[index] = gammaDistribution(generator);
            beta[index] = betaDistribution(generator);
        }
        const std::vector<float> rmsnormReference = referenceRmsnorm(input, gamma, beta);
        const std::vector<float> layernormReference = referenceLayernorm(input);
        const std::vector<float> fusedReference =
            referenceFusedResidualRmsnorm(input, residual, gamma);

        const std::size_t floatBytes = static_cast<std::size_t>(hiddenSize) * sizeof(float);
        float *deviceInput = nullptr, *deviceResidual = nullptr, *deviceOutput = nullptr;
        float *deviceGamma = nullptr, *deviceBeta = nullptr;
        checkCuda(cudaMalloc(&deviceInput, floatBytes), "allocate float input");
        checkCuda(cudaMalloc(&deviceResidual, floatBytes), "allocate float residual");
        checkCuda(cudaMalloc(&deviceOutput, floatBytes), "allocate float output");
        checkCuda(cudaMalloc(&deviceGamma, floatBytes), "allocate float gamma");
        checkCuda(cudaMalloc(&deviceBeta, floatBytes), "allocate float beta");
        checkCuda(cudaMemcpy(deviceInput, input.data(), floatBytes, cudaMemcpyHostToDevice), "copy float input");
        checkCuda(cudaMemcpy(deviceResidual, residual.data(), floatBytes, cudaMemcpyHostToDevice), "copy float residual");
        checkCuda(cudaMemcpy(deviceGamma, gamma.data(), floatBytes, cudaMemcpyHostToDevice), "copy float gamma");
        checkCuda(cudaMemcpy(deviceBeta, beta.data(), floatBytes, cudaMemcpyHostToDevice), "copy float beta");

        auto reportFloat = [&](const char* kernel, const std::vector<float>& expected,
                               double transferredArrays, auto launch) {
            const auto [averageMs, minimumMs] = timeKernel(iterations, warmupIterations, launch);
            std::vector<float> result(hiddenSize);
            checkCuda(cudaMemcpy(result.data(), deviceOutput, floatBytes, cudaMemcpyDeviceToHost), "copy float output");
            const ErrorMetrics error = measureError(result, expected, floatAtol, floatRtol);
            const double bandwidth = transferredArrays * floatBytes / (averageMs * 1.0e6);
            const bool passed = error.errorCount == 0;
            std::printf("%s,float32,%d,%d,%d,%.6f,%.6f,%.3f,%.6e,%.6e,%.6e,%zu,%s\n",
                        kernel, hiddenSize, blockSize, iterations, averageMs, minimumMs, bandwidth,
                        error.maxAbs, error.maxRel, error.rmse, error.errorCount, passed ? "PASS" : "FAIL");
            if (!passed) ++failedRuns;
        };
        reportFloat("naive", rmsnormReference, 4.0, [&] { launch_naive_rmsnorm(deviceInput, deviceOutput, deviceGamma, deviceBeta, hiddenSize, blockSize); });
        reportFloat("warp", rmsnormReference, 4.0, [&] { launch_warp_rmsnorm(deviceInput, deviceOutput, deviceGamma, deviceBeta, hiddenSize, blockSize); });
        reportFloat("float4", rmsnormReference, 4.0, [&] { launch_vectorized_rmsnorm(deviceInput, deviceOutput, deviceGamma, deviceBeta, hiddenSize, blockSize); });
        reportFloat("layernorm", layernormReference, 3.0, [&] { launch_layernorm(deviceInput, deviceOutput, hiddenSize, blockSize); });
        reportFloat("welford", layernormReference, 2.0, [&] { launch_welford_layernorm(deviceInput, deviceOutput, hiddenSize, blockSize); });
        reportFloat("fused", fusedReference, 4.0, [&] { launch_fused_residual_rmsnorm(deviceInput, deviceResidual, deviceGamma, deviceOutput, hiddenSize, blockSize); });

        std::vector<half> inputHalf(hiddenSize), gammaHalf(hiddenSize), betaHalf(hiddenSize), outputHalf(hiddenSize);
        for (int index = 0; index < hiddenSize; ++index) {
            inputHalf[index] = __float2half(input[index]);
            gammaHalf[index] = __float2half(gamma[index]);
            betaHalf[index] = __float2half(beta[index]);
        }
        const std::size_t halfBytes = static_cast<std::size_t>(hiddenSize) * sizeof(half);
        half *deviceInputHalf = nullptr, *deviceOutputHalf = nullptr, *deviceGammaHalf = nullptr, *deviceBetaHalf = nullptr;
        checkCuda(cudaMalloc(&deviceInputHalf, halfBytes), "allocate half input");
        checkCuda(cudaMalloc(&deviceOutputHalf, halfBytes), "allocate half output");
        checkCuda(cudaMalloc(&deviceGammaHalf, halfBytes), "allocate half gamma");
        checkCuda(cudaMalloc(&deviceBetaHalf, halfBytes), "allocate half beta");
        checkCuda(cudaMemcpy(deviceInputHalf, inputHalf.data(), halfBytes, cudaMemcpyHostToDevice), "copy half input");
        checkCuda(cudaMemcpy(deviceGammaHalf, gammaHalf.data(), halfBytes, cudaMemcpyHostToDevice), "copy half gamma");
        checkCuda(cudaMemcpy(deviceBetaHalf, betaHalf.data(), halfBytes, cudaMemcpyHostToDevice), "copy half beta");
        std::vector<float> halfResult(hiddenSize);
        auto reportHalf = [&](const char* kernel, auto launch) {
            const auto [averageMs, minimumMs] = timeKernel(iterations, warmupIterations, launch);
            checkCuda(cudaMemcpy(outputHalf.data(), deviceOutputHalf, halfBytes, cudaMemcpyDeviceToHost), "copy half output");
            std::transform(outputHalf.begin(), outputHalf.end(), halfResult.begin(), [](half value) { return __half2float(value); });
            const ErrorMetrics error = measureError(halfResult, rmsnormReference, halfAtol, halfRtol);
            const double bandwidth = 4.0 * halfBytes / (averageMs * 1.0e6);
            const bool passed = error.errorCount == 0;
            std::printf("%s,float16,%d,%d,%d,%.6f,%.6f,%.3f,%.6e,%.6e,%.6e,%zu,%s\n",
                        kernel, hiddenSize, blockSize, iterations, averageMs, minimumMs, bandwidth,
                        error.maxAbs, error.maxRel, error.rmse, error.errorCount, passed ? "PASS" : "FAIL");
            if (!passed) ++failedRuns;
        };
        reportHalf("warp_half", [&] {
            launch_warp_rmsnorm_half(deviceInputHalf, deviceOutputHalf, deviceGammaHalf,
                                     deviceBetaHalf, hiddenSize, blockSize);
        });
        reportHalf("half2", [&] {
            launch_vectorized_rmsnorm_half2(deviceInputHalf, deviceOutputHalf, deviceGammaHalf,
                                            deviceBetaHalf, hiddenSize, blockSize);
        });

        cudaFree(deviceInput); cudaFree(deviceResidual); cudaFree(deviceOutput); cudaFree(deviceGamma); cudaFree(deviceBeta);
        cudaFree(deviceInputHalf); cudaFree(deviceOutputHalf); cudaFree(deviceGammaHalf); cudaFree(deviceBetaHalf);
    }

    if (failedRuns != 0) {
        std::fprintf(stderr, "ERROR REPORT: %d benchmark run(s) exceeded precision-specific tolerances\n", failedRuns);
        return EXIT_FAILURE;
    }
    std::fprintf(stderr, "ERROR REPORT: all runs passed float(atol=%.1e, rtol=%.1e), half(atol=%.1e, rtol=%.1e)\n",
                 floatAtol, floatRtol, halfAtol, halfRtol);
    return EXIT_SUCCESS;
}
