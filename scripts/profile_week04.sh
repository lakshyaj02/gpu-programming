#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

kernel="${1:-shared}"
M="${2:-1024}"
N="${3:-1024}"
K="${4:-1024}"
tile_size="${5:-16}"

if [[ "$kernel" != "naive" && "$kernel" != "shared" && "$kernel" != "thread_coarse" && "$kernel" != "cublas" ]]; then
	printf 'error: kernel must be naive, shared, thread_coarse, or cublas\n' >&2
	exit 1
fi

for value in "$M" "$N" "$K" "$tile_size"; do
	if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
		printf 'error: dimensions and tile size must be positive integers\n' >&2
		exit 1
	fi
done

if [[ "$kernel" != "cublas" && "$tile_size" != "8" && "$tile_size" != "16" && "$tile_size" != "32" ]]; then
	printf 'error: tile size must be 8, 16, or 32 for custom kernels\n' >&2
	exit 1
fi

for command in cmake ncu; do
	if ! command -v "$command" >/dev/null 2>&1; then
		printf 'error: required command not found: %s\n' "$command" >&2
		exit 1
	fi
done

build_dir="week04-gemm/build"
results_dir="week04-gemm/results/profiles"
mkdir -p "$results_dir"

printf '%s\n' 'Configuring and building week04_profile...'
cmake -S week04-gemm -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" -j --target week04_profile

timestamp="$(date +%Y%m%d_%H%M%S)"
profile_name="${kernel}_M${M}_N${N}_K${K}_tile${tile_size}_${timestamp}"
report_base="$results_dir/$profile_name"
csv_path="${report_base}.csv"

printf 'Profiling %s GEMM M=%s N=%s K=%s tile=%s...\n' "$kernel" "$M" "$N" "$K" "$tile_size"
ncu \
	--profile-from-start off \
	--target-processes all \
	--section SpeedOfLight \
	--section MemoryWorkloadAnalysis \
	--section SchedulerStats \
	--section Occupancy \
	--section LaunchStats \
	--section WarpStateStats \
	--section InstructionStats \
	--page raw \
	--csv \
	--log-file "$csv_path" \
	--export "$report_base" \
	--force-overwrite \
	"$build_dir/week04_profile" "$kernel" "$M" "$N" "$K" "$tile_size"

printf 'Nsight Compute report written to %s.ncu-rep\n' "$report_base"
printf 'Raw metrics written to %s\n' "$csv_path"