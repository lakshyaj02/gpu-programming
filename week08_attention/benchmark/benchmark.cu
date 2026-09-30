#include "attention_kernels.cuh"
#include "cuda_check.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

namespace {

std::vector<float> referenceAttention(const std::vector<float>& q,
                                      const std::vector<float>& k,
                                      const std::vector<float>& v,
                                      int m, int n, int d, int dv,
                                      float scale, bool causal) {
    std::vector<float> probabilities(static_cast<size_t>(m) * n);
    std::vector<float> output(static_cast<size_t>(m) * dv, 0.0f);

    for (int row = 0; row < m; ++row) {
        float maximum = -INFINITY;
        for (int col = 0; col < n; ++col) {
            float score = 0.0f;
            for (int inner = 0; inner < d; ++inner) {
                score += q[row * d + inner] * k[col * d + inner];
            }
            score = causal && col > row ? -1e9f : score * scale;
            probabilities[row * n + col] = score;
            maximum = std::max(maximum, score);
        }

        float sum = 0.0f;
        for (int col = 0; col < n; ++col) {
            float value = std::exp(probabilities[row * n + col] - maximum);
            probabilities[row * n + col] = value;
            sum += value;
        }
        for (int col = 0; col < n; ++col) {
            probabilities[row * n + col] /= sum;
        }
    }

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < dv; ++col) {
            for (int inner = 0; inner < n; ++inner) {
                output[row * dv + col] +=
                    probabilities[row * n + inner] * v[inner * dv + col];
            }
        }
    }
    return output;
}

bool approximatelyEqual(const std::vector<float>& actual,
                        const std::vector<float>& expected) {
    for (size_t index = 0; index < actual.size(); ++index) {
        float tolerance = 2e-4f + 2e-3f * std::abs(expected[index]);
        if (!std::isfinite(actual[index]) ||
            std::abs(actual[index] - expected[index]) > tolerance) {
            std::cerr << "Mismatch at " << index << ": expected "
                      << expected[index] << ", got " << actual[index] << '\n';
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int m = argc > 1 ? std::atoi(argv[1]) : 128;
    int n = argc > 2 ? std::atoi(argv[2]) : 128;
    int d = argc > 3 ? std::atoi(argv[3]) : 64;
    int dv = argc > 4 ? std::atoi(argv[4]) : 64;
    bool causal = argc > 5 && std::atoi(argv[5]) != 0;
    if (m <= 0 || n <= 0 || d <= 0 || dv <= 0) {
        std::cerr << "All dimensions must be positive\n";
        return EXIT_FAILURE;
    }

    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> q(static_cast<size_t>(m) * d);
    std::vector<float> k(static_cast<size_t>(n) * d);
    std::vector<float> v(static_cast<size_t>(n) * dv);
    std::generate(q.begin(), q.end(), [&] { return distribution(generator); });
    std::generate(k.begin(), k.end(), [&] { return distribution(generator); });
    std::generate(v.begin(), v.end(), [&] { return distribution(generator); });

    float* device_q = nullptr;
    float* device_k = nullptr;
    float* device_v = nullptr;
    float* device_output = nullptr;
    CUDA_CHECK(cudaMalloc(&device_q, q.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_k, k.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_v, v.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_output, static_cast<size_t>(m) * dv * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_k, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_v, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));

    float scale = 1.0f / std::sqrt(static_cast<float>(d));
    launch_attention_tiled(device_q, device_k, device_v, device_output,
                           m, n, d, dv, scale, causal);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> actual(static_cast<size_t>(m) * dv);
    CUDA_CHECK(cudaMemcpy(actual.data(), device_output,
                          actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> expected = referenceAttention(q, k, v, m, n, d, dv,
                                                     scale, causal);
    bool correct = approximatelyEqual(actual, expected);

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    constexpr int iterations = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        launch_attention_tiled(device_q, device_k, device_v, device_output,
                               m, n, d, dv, scale, causal);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    std::cout << "shape=" << m << 'x' << n << " d=" << d << " dv=" << dv
              << " causal=" << causal << " latency_ms=" << elapsed_ms / iterations
              << " correctness=" << (correct ? "PASS" : "FAIL") << '\n';

    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_v));
    CUDA_CHECK(cudaFree(device_k));
    CUDA_CHECK(cudaFree(device_q));
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
