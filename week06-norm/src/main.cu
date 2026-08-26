#include "norm_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

void cudaCheck(cudaError_t status, const char* operation) {
	if (status != cudaSuccess) {
		std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
		std::exit(EXIT_FAILURE);
	}
}

std::vector<float> rmsnormReference(const std::vector<float>& input,
									const std::vector<float>& gamma,
									const std::vector<float>& beta) {
	double square_sum = 0.0;
	for (float value : input) {
		square_sum += static_cast<double>(value) * value;
	}
	const float inverse_rms = 1.0f / std::sqrt(
		static_cast<float>(square_sum / input.size()) + 1e-6f);

	std::vector<float> output(input.size());
	for (std::size_t index = 0; index < input.size(); ++index) {
		output[index] = input[index] * inverse_rms * gamma[index] + beta[index];
	}
	return output;
}

std::vector<float> layernormReference(const std::vector<float>& input) {
	double sum = 0.0;
	for (float value : input) {
		sum += value;
	}
	const double mean = sum / input.size();

	double squared_difference_sum = 0.0;
	for (float value : input) {
		const double difference = value - mean;
		squared_difference_sum += difference * difference;
	}
	const double inverse_standard_deviation =
		1.0 / std::sqrt(squared_difference_sum / input.size() + 1e-5);

	std::vector<float> output(input.size());
	for (std::size_t index = 0; index < input.size(); ++index) {
		output[index] = static_cast<float>((input[index] - mean) * inverse_standard_deviation);
	}
	return output;
}

std::vector<float> fusedResidualRmsnormReference(const std::vector<float>& input,
												 const std::vector<float>& residual,
												 const std::vector<float>& gamma) {
	double square_sum = 0.0;
	for (std::size_t index = 0; index < input.size(); ++index) {
		const double value = static_cast<double>(input[index]) + residual[index];
		square_sum += value * value;
	}
	const double inverse_rms = 1.0 / std::sqrt(square_sum / input.size() + 1e-6);
	std::vector<float> output(input.size());
	for (std::size_t index = 0; index < input.size(); ++index) {
		output[index] = static_cast<float>((input[index] + residual[index]) *
										 inverse_rms * gamma[index]);
	}
	return output;
}

struct ErrorMetrics {
	float max_absolute = 0.0f;
	float max_relative = 0.0f;
	double root_mean_square = 0.0;
};

ErrorMetrics measureError(const std::vector<float>& actual,
						  const std::vector<float>& expected) {
	ErrorMetrics metrics;
	double square_error_sum = 0.0;
	for (std::size_t index = 0; index < actual.size(); ++index) {
		const float absolute = std::abs(actual[index] - expected[index]);
		const float relative = absolute / std::max(std::abs(expected[index]), 1e-6f);
		metrics.max_absolute = std::max(metrics.max_absolute, absolute);
		metrics.max_relative = std::max(metrics.max_relative, relative);
		square_error_sum += static_cast<double>(absolute) * absolute;
	}
	metrics.root_mean_square = std::sqrt(square_error_sum / actual.size());
	return metrics;
}

template <typename Launch>
float benchmarkKernel(Launch launch, int warmup_iterations = 10, int iterations = 100) {
	for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
		launch();
	}
	cudaCheck(cudaDeviceSynchronize(), "kernel warmup");

	cudaEvent_t start;
	cudaEvent_t stop;
	cudaCheck(cudaEventCreate(&start), "create start event");
	cudaCheck(cudaEventCreate(&stop), "create stop event");
	cudaCheck(cudaEventRecord(start), "record start event");
	for (int iteration = 0; iteration < iterations; ++iteration) {
		launch();
	}
	cudaCheck(cudaEventRecord(stop), "record stop event");
	cudaCheck(cudaEventSynchronize(stop), "synchronize stop event");

	float elapsed_milliseconds = 0.0f;
	cudaCheck(cudaEventElapsedTime(&elapsed_milliseconds, start, stop), "measure kernel time");
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return elapsed_milliseconds / iterations;
}

void printResult(const char* kernel, int hidden_size, float average_milliseconds,
				 const ErrorMetrics& error) {
	std::printf("%-12s | %6d | %10.6f | %11.3e | %11.3e | %11.3e\n",
				kernel, hidden_size, average_milliseconds, error.max_absolute,
				error.max_relative, error.root_mean_square);
}

void runSize(int hidden_size) {
	std::mt19937 generator(42 + hidden_size);
	std::uniform_real_distribution<float> input_distribution(-2.0f, 2.0f);
	std::uniform_real_distribution<float> gamma_distribution(0.5f, 1.5f);
	std::uniform_real_distribution<float> beta_distribution(-0.25f, 0.25f);

	std::vector<float> input(hidden_size);
	std::vector<float> residual(hidden_size);
	std::vector<float> gamma(hidden_size);
	std::vector<float> beta(hidden_size);
	for (int index = 0; index < hidden_size; ++index) {
		input[index] = input_distribution(generator);
		residual[index] = input_distribution(generator);
		gamma[index] = gamma_distribution(generator);
		beta[index] = beta_distribution(generator);
	}
	const std::vector<float> reference = rmsnormReference(input, gamma, beta);
	const std::vector<float> layernorm_reference = layernormReference(input);
	const std::vector<float> fused_reference =
		fusedResidualRmsnormReference(input, residual, gamma);

	std::vector<half> input_half(hidden_size);
	std::vector<half> gamma_half(hidden_size);
	std::vector<half> beta_half(hidden_size);
	for (int index = 0; index < hidden_size; ++index) {
		input_half[index] = __float2half(input[index]);
		gamma_half[index] = __float2half(gamma[index]);
		beta_half[index] = __float2half(beta[index]);
	}

	float *device_input, *device_residual, *device_output, *device_gamma, *device_beta;
	half *device_input_half, *device_output_half, *device_gamma_half, *device_beta_half;
	const std::size_t float_bytes = hidden_size * sizeof(float);
	const std::size_t half_bytes = hidden_size * sizeof(half);
	cudaCheck(cudaMalloc(&device_input, float_bytes), "cudaMalloc float input");
	cudaCheck(cudaMalloc(&device_residual, float_bytes), "cudaMalloc float residual");
	cudaCheck(cudaMalloc(&device_output, float_bytes), "cudaMalloc float output");
	cudaCheck(cudaMalloc(&device_gamma, float_bytes), "cudaMalloc float gamma");
	cudaCheck(cudaMalloc(&device_beta, float_bytes), "cudaMalloc float beta");
	cudaCheck(cudaMalloc(&device_input_half, half_bytes), "cudaMalloc half input");
	cudaCheck(cudaMalloc(&device_output_half, half_bytes), "cudaMalloc half output");
	cudaCheck(cudaMalloc(&device_gamma_half, half_bytes), "cudaMalloc half gamma");
	cudaCheck(cudaMalloc(&device_beta_half, half_bytes), "cudaMalloc half beta");

	cudaCheck(cudaMemcpy(device_input, input.data(), float_bytes, cudaMemcpyHostToDevice), "copy float input");
	cudaCheck(cudaMemcpy(device_residual, residual.data(), float_bytes, cudaMemcpyHostToDevice), "copy float residual");
	cudaCheck(cudaMemcpy(device_gamma, gamma.data(), float_bytes, cudaMemcpyHostToDevice), "copy float gamma");
	cudaCheck(cudaMemcpy(device_beta, beta.data(), float_bytes, cudaMemcpyHostToDevice), "copy float beta");
	cudaCheck(cudaMemcpy(device_input_half, input_half.data(), half_bytes, cudaMemcpyHostToDevice), "copy half input");
	cudaCheck(cudaMemcpy(device_gamma_half, gamma_half.data(), half_bytes, cudaMemcpyHostToDevice), "copy half gamma");
	cudaCheck(cudaMemcpy(device_beta_half, beta_half.data(), half_bytes, cudaMemcpyHostToDevice), "copy half beta");

	std::vector<float> float_output(hidden_size);
	std::vector<half> half_output(hidden_size);
	std::vector<float> half_output_float(hidden_size);

	auto run_float = [&](const char* kernel, auto launch) {
		const float average_milliseconds = benchmarkKernel(launch);
		cudaCheck(cudaGetLastError(), kernel);
		cudaCheck(cudaMemcpy(float_output.data(), device_output, float_bytes, cudaMemcpyDeviceToHost), "copy float output");
		printResult(kernel, hidden_size, average_milliseconds,
					measureError(float_output, reference));
	};

	auto run_half = [&](const char* kernel, auto launch) {
		const float average_milliseconds = benchmarkKernel(launch);
		cudaCheck(cudaGetLastError(), kernel);
		cudaCheck(cudaMemcpy(half_output.data(), device_output_half, half_bytes, cudaMemcpyDeviceToHost), "copy half output");
		std::transform(half_output.begin(), half_output.end(), half_output_float.begin(),
					   [](half value) { return __half2float(value); });
		printResult(kernel, hidden_size, average_milliseconds,
					measureError(half_output_float, reference));
	};

	run_float("naive", [&] {
		launch_naive_rmsnorm(device_input, device_output, device_gamma, device_beta,
							 hidden_size, 256);
	});
	run_float("warp", [&] {
		launch_warp_rmsnorm(device_input, device_output, device_gamma, device_beta,
							hidden_size, 256);
	});
	run_float("float4", [&] {
		launch_vectorized_rmsnorm(device_input, device_output, device_gamma, device_beta,
								  hidden_size, 256);
	});
	run_half("warp_half", [&] {
		launch_warp_rmsnorm_half(device_input_half, device_output_half, device_gamma_half,
							 device_beta_half, hidden_size, 256);
	});
	run_half("half2", [&] {
		launch_vectorized_rmsnorm_half2(device_input_half, device_output_half,
									device_gamma_half, device_beta_half, hidden_size, 256);
	});

	auto run_layernorm = [&](const char* kernel, auto launch) {
		const float average_milliseconds = benchmarkKernel(launch);
		cudaCheck(cudaGetLastError(), kernel);
		cudaCheck(cudaMemcpy(float_output.data(), device_output, float_bytes,
							 cudaMemcpyDeviceToHost), "copy LayerNorm output");
		printResult(kernel, hidden_size, average_milliseconds,
					measureError(float_output, layernorm_reference));
	};

	run_layernorm("layernorm", [&] {
		launch_layernorm(device_input, device_output, hidden_size, 256);
	});
	run_layernorm("welford", [&] {
		launch_welford_layernorm(device_input, device_output, hidden_size, 256);
	});

	const float fused_milliseconds = benchmarkKernel([&] {
		launch_fused_residual_rmsnorm(device_input, device_residual, device_gamma,
									 device_output, hidden_size, 256);
	});
	cudaCheck(cudaGetLastError(), "fused_residual");
	cudaCheck(cudaMemcpy(float_output.data(), device_output, float_bytes,
						 cudaMemcpyDeviceToHost), "copy fused residual output");
	printResult("fused", hidden_size, fused_milliseconds,
				measureError(float_output, fused_reference));

	cudaFree(device_input);
	cudaFree(device_residual);
	cudaFree(device_output);
	cudaFree(device_gamma);
	cudaFree(device_beta);
	cudaFree(device_input_half);
	cudaFree(device_output_half);
	cudaFree(device_gamma_half);
	cudaFree(device_beta_half);
}

}  // namespace

int main() {
	std::puts("kernel       |   size |     avg ms |     max abs |     max rel |        RMSE");
	std::puts("-------------+--------+------------+-------------+-------------+------------");
	for (int hidden_size : {768, 1024, 2048, 4096, 8192}) {
		runSize(hidden_size);
	}
	return 0;
}
