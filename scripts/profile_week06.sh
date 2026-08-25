#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

kernel="${1:-warp}"
hidden_size="${2:-4096}"
block_size="${3:-256}"
if [[ "$kernel" != "naive" && "$kernel" != "warp" && "$kernel" != "warp_half" ]]; then
	printf 'error: kernel must be naive, warp, or warp_half\n' >&2
	exit 1
fi
for value in "$hidden_size" "$block_size"; do
	if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
		printf 'error: hidden size and block size must be positive integers\n' >&2
		exit 1
	fi
done
if (( block_size < 32 || block_size > 1024 || block_size % 32 != 0 )); then
	printf 'error: block size must be a multiple of 32 from 32 through 1024\n' >&2
	exit 1
fi
for command in cmake ncu; do
	if ! command -v "$command" >/dev/null 2>&1; then
		printf 'error: required command not found: %s\n' "$command" >&2
		exit 1
	fi
done

build_dir="week06-norm/build"
results_dir="week06-norm/results/profiles"
mkdir -p "$results_dir"
printf '%s\n' 'Configuring and building week06_profile...'
cmake -S week06-norm -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" -j --target week06_profile

timestamp="$(date +%Y%m%d_%H%M%S)"
profile_name="${kernel}_H${hidden_size}_block${block_size}_${timestamp}"
report_base="$results_dir/$profile_name"
csv_path="${report_base}.csv"
printf 'Profiling %s RMSNorm H=%s block=%s...\n' "$kernel" "$hidden_size" "$block_size"
ncu \
	--profile-from-start off \
	--target-processes all \
	--section SpeedOfLight \
	--section MemoryWorkloadAnalysis \
	--section SchedulerStats \
	--section Occupancy \
	--section LaunchStats \
	--section WarpStateStats \
	--page raw \
	--csv \
	--log-file "$csv_path" \
	--export "$report_base" \
	--force-overwrite \
	"$build_dir/week06_profile" "$kernel" "$hidden_size" "$block_size"

printf 'Nsight Compute report written to %s.ncu-rep\n' "$report_base"
printf 'Raw metrics written to %s\n' "$csv_path"