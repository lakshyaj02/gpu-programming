#pragma once

#include <chrono>

class CpuTimer {
public:
	void start() {
		startTime_ = Clock::now();
	}

	float stop() const {
		const auto elapsed = Clock::now() - startTime_;
		return std::chrono::duration<float, std::milli>(elapsed).count();
	}

private:
	using Clock = std::chrono::steady_clock;
	Clock::time_point startTime_{};
};
