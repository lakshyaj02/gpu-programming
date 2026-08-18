#include "common.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {

float quantizeFp8E4m3(float value) {
	const float sign = std::copysign(1.0f, value);
	const float magnitude = std::min(std::fabs(value), 448.0f);
	if (magnitude < 0.015625f) {
		return sign * std::round(magnitude * 512.0f) / 512.0f;
	}
	const int exponent = static_cast<int>(std::floor(std::log2(magnitude)));
	const float quantum = std::ldexp(1.0f, exponent - 3);
	return sign * std::min(std::round(magnitude / quantum) * quantum, 448.0f);
}

float quantizeFp4E2m1(float value) {
	const float sign = std::copysign(1.0f, value);
	const float magnitude = std::min(std::fabs(value), 6.0f);
	if (magnitude < 1.0f) {
		return sign * std::round(magnitude * 2.0f) / 2.0f;
	}
	const int exponent = static_cast<int>(std::floor(std::log2(magnitude)));
	const float quantum = std::ldexp(1.0f, exponent - 1);
	return sign * std::min(std::round(magnitude / quantum) * quantum, 6.0f);
}

}  // namespace

void runPrecisionDemo() {
	constexpr float value = 0.1f;
	const float bf16Value = __bfloat162float(__float2bfloat16(value));
	std::fprintf(stderr, "Precision demo (stored representation of %.8f):\n", value);
	std::fprintf(stderr, "  FP32:     %.8f\n", value);
	std::fprintf(stderr, "  BF16:     %.8f\n", bf16Value);
	std::fprintf(stderr, "  FP8 E4M3: %.8f\n", quantizeFp8E4m3(value));
	std::fprintf(stderr, "  FP4 E2M1: %.8f\n", quantizeFp4E2m1(value));
}