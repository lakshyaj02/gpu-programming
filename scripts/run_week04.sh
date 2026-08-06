#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

iterations="20"
warmup_iterations="5"
run_profile=false
profile_args=()
positional_args=()

while [[ "$#" -gt 0 ]]; do
	case "$1" in
		--profile)
			run_profile=true
			shift
			while [[ "$#" -gt 0 ]]; do
				profile_args+=("$1")
				shift
			done
			;;
		-h|--help)
			printf 'Usage: %s [iterations] [warmup_iterations] [--profile [kernel M N K tile_size]]\n' "$0"
			exit 0
			;;
		*)
			positional_args+=("$1")
			shift
			;;
	esac
done

if [[ "${#positional_args[@]}" -gt 2 ]]; then
	printf 'error: expected at most iterations and warmup_iterations before --profile\n' >&2
	exit 1
fi

if [[ "${#positional_args[@]}" -ge 1 ]]; then
	iterations="${positional_args[0]}"
fi
if [[ "${#positional_args[@]}" -eq 2 ]]; then
	warmup_iterations="${positional_args[1]}"
fi

for value in "$iterations" "$warmup_iterations"; do
	if [[ ! "$value" =~ ^[0-9]+$ ]]; then
		printf 'error: iterations and warmup_iterations must be integers\n' >&2
		exit 1
	fi
done
if [[ "$iterations" == "0" ]]; then
	printf 'error: iterations must be greater than zero\n' >&2
	exit 1
fi

for command in cmake nvcc; do
	if ! command -v "$command" >/dev/null 2>&1; then
		printf 'error: required command not found: %s\n' "$command" >&2
		exit 1
	fi
done

build_dir="week04-gemm/build"
results_dir="week04-gemm/results/raw"
mkdir -p "$results_dir"

timestamp="$(date +%Y%m%d_%H%M%S)"
cuda_csv="$results_dir/cuda_timings_gemm_${timestamp}.csv"
error_log="$results_dir/error_report_gemm_${timestamp}.log"

printf '%s\n' 'Configuring and building week04_benchmark...'
cmake -S week04-gemm -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" -j --target week04_benchmark

printf '%s\n' 'Running CUDA and cuBLAS benchmark harness...'
if ! "$build_dir/week04_benchmark" "$iterations" "$warmup_iterations" \
	> "$cuda_csv" 2> "$error_log"; then
	printf 'error: benchmark correctness checks failed; see %s\n' "$error_log" >&2
	exit 1
fi

printf 'CUDA timings written to %s\n' "$cuda_csv"
printf 'Error report written to %s\n' "$error_log"

if [[ "$run_profile" == true ]]; then
	printf '%s\n' 'Running focused Nsight Compute profile...'
	"$project_root/scripts/profile_week04.sh" "${profile_args[@]}"
fi

printf 'Done. Week 4 benchmark artifacts written under %s\n' "$results_dir"
