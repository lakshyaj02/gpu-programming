#include "fused_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

float gelu(float value) {
    return 0.5f * value * (1.0f + std::tanh(0.7978845608f *
           (value + 0.044715f * value * value * value)));
}

bool check(const char* name, const std::vector<float>& actual,
           const std::vector<float>& expected, float tolerance = 2.0e-4f) {
    float maximum_error = 0.0f;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        maximum_error = std::max(maximum_error, std::abs(actual[index] - expected[index]));
    }
    const bool passed = maximum_error <= tolerance;
    std::printf("%-24s max error %.3e  %s\n", name, maximum_error, passed ? "PASS" : "FAIL");
    return passed;
}

}  // namespace

int main() {
    constexpr int rows = 7;
    constexpr int columns = 513;
    const int count = rows * columns;
    const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(float);
    std::mt19937 generator(7);
    std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);
    std::vector<float> input(count), residual(count), gamma(columns), beta(columns);
    for (float& value : input) value = distribution(generator);
    for (float& value : residual) value = distribution(generator);
    for (float& value : gamma) value = 1.0f + 0.1f * distribution(generator);
    for (float& value : beta) value = 0.1f * distribution(generator);

    float *device_input, *device_residual, *device_gamma, *device_beta;
    float *device_scratch, *device_stats, *device_output;
    checkCuda(cudaMalloc(&device_input, bytes), "allocate input");
    checkCuda(cudaMalloc(&device_residual, bytes), "allocate residual");
    checkCuda(cudaMalloc(&device_gamma, columns * sizeof(float)), "allocate gamma");
    checkCuda(cudaMalloc(&device_beta, columns * sizeof(float)), "allocate beta");
    checkCuda(cudaMalloc(&device_scratch, bytes), "allocate scratch");
    checkCuda(cudaMalloc(&device_stats, 2 * rows * sizeof(float)), "allocate stats");
    checkCuda(cudaMalloc(&device_output, bytes), "allocate output");
    checkCuda(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice), "copy input");
    checkCuda(cudaMemcpy(device_residual, residual.data(), bytes, cudaMemcpyHostToDevice), "copy residual");
    checkCuda(cudaMemcpy(device_gamma, gamma.data(), columns * sizeof(float), cudaMemcpyHostToDevice), "copy gamma");
    checkCuda(cudaMemcpy(device_beta, beta.data(), columns * sizeof(float), cudaMemcpyHostToDevice), "copy beta");

    std::vector<float> expected(count), actual(count);
    bool passed = true;
    auto validate = [&](const char* name, auto launch, float tolerance = 2.0e-4f) {
        launch();
        checkCuda(cudaMemcpy(actual.data(), device_output, bytes, cudaMemcpyDeviceToHost), "copy output");
        passed &= check(name, actual, expected, tolerance);
    };

    for (int index = 0; index < count; ++index) expected[index] = std::max(input[index] + beta[index % columns], 0.0f);
    validate("7.1 unfused bias + ReLU", [&] { launch_bias_relu_unfused(device_input, device_beta, device_scratch, device_output, rows, columns); });
    validate("7.1 fused bias + ReLU", [&] { launch_bias_relu_fused(device_input, device_beta, device_output, rows, columns); });

    for (int index = 0; index < count; ++index) expected[index] = gelu(input[index] + beta[index % columns]);
    validate("7.2 unfused bias + GELU", [&] { launch_bias_gelu_unfused(device_input, device_beta, device_scratch, device_output, rows, columns); });
    validate("7.2 fused bias + GELU", [&] { launch_bias_gelu_fused(device_input, device_beta, device_output, rows, columns); });

    for (int row = 0; row < rows; ++row) {
        double squares = 0.0;
        for (int column = 0; column < columns; ++column) squares += static_cast<double>(input[row * columns + column]) * input[row * columns + column];
        const double inverse_rms = 1.0 / std::sqrt(squares / columns + 1.0e-6);
        for (int column = 0; column < columns; ++column) expected[row * columns + column] = static_cast<float>(input[row * columns + column] * inverse_rms * gamma[column]);
    }
    validate("7.3 unfused RMSNorm", [&] { launch_rmsnorm_unfused(device_input, device_gamma, device_scratch, device_stats, device_output, rows, columns); });
    validate("7.3 fused RMSNorm", [&] { launch_rmsnorm_fused(device_input, device_gamma, device_output, rows, columns); });

    for (int row = 0; row < rows; ++row) {
        double sum = 0.0, squares = 0.0;
        for (int column = 0; column < columns; ++column) { const double value = input[row * columns + column]; sum += value; squares += value * value; }
        const double mean = sum / columns;
        const double inverse_std = 1.0 / std::sqrt(std::max(squares / columns - mean * mean, 0.0) + 1.0e-5);
        for (int column = 0; column < columns; ++column) expected[row * columns + column] = static_cast<float>((input[row * columns + column] - mean) * inverse_std * gamma[column] + beta[column]);
    }
    validate("7.4 unfused LayerNorm", [&] { launch_layernorm_unfused(device_input, device_gamma, device_beta, device_scratch, device_stats, device_output, rows, columns); });
    validate("7.4 fused LayerNorm", [&] { launch_layernorm_fused(device_input, device_gamma, device_beta, device_output, rows, columns); });

    for (int row = 0; row < rows; ++row) {
        double squares = 0.0;
        for (int column = 0; column < columns; ++column) { const double value = input[row * columns + column] + residual[row * columns + column]; squares += value * value; }
        const double inverse_rms = 1.0 / std::sqrt(squares / columns + 1.0e-6);
        for (int column = 0; column < columns; ++column) expected[row * columns + column] = static_cast<float>((input[row * columns + column] + residual[row * columns + column]) * inverse_rms * gamma[column]);
    }
    validate("7.5 unfused residual RMS", [&] { launch_residual_rmsnorm_unfused(device_input, device_residual, device_gamma, device_scratch, device_stats, device_output, rows, columns); });
    validate("7.5 fused residual RMS", [&] { launch_residual_rmsnorm_fused(device_input, device_residual, device_gamma, device_output, rows, columns); });

    for (int row = 0; row < rows; ++row) {
        float maximum = input[row * columns];
        for (int column = 1; column < columns; ++column) maximum = std::max(maximum, input[row * columns + column]);
        double sum = 0.0;
        for (int column = 0; column < columns; ++column) sum += std::exp(input[row * columns + column] - maximum);
        for (int column = 0; column < columns; ++column) expected[row * columns + column] = static_cast<float>(std::exp(input[row * columns + column] - maximum) / sum);
    }
    validate("7.6 unfused softmax", [&] { launch_softmax_unfused(device_input, device_scratch, device_stats, device_output, rows, columns); });
    validate("7.6 fused softmax", [&] { launch_softmax_fused(device_input, device_output, rows, columns); });

    cudaFree(device_input); cudaFree(device_residual); cudaFree(device_gamma); cudaFree(device_beta);
    cudaFree(device_scratch); cudaFree(device_stats); cudaFree(device_output);
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}