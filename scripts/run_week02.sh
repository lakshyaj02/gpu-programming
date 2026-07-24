#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

for command in cmake nvcc; do
	if ! command -v "$command" >/dev/null 2>&1; then
		printf 'error: required command not found: %s\n' "$command" >&2
		exit 1
	fi
done

results_dir="week02-memory-tiling/results/raw_out"
mkdir -p "$results_dir"

timestamp="$(date +%Y%m%d_%H%M%S)"
out_file="$results_dir/week02_benchmark_${timestamp}.csv"

printf '%s\n' 'Configuring and building week02_benchmark...'
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target week02_benchmark

printf 'Running week02_benchmark; output will be saved to %s\n' "$out_file"
./build/week02_benchmark "$@" | tee "$out_file"

printf 'Done. Saved benchmark output to %s\n' "$out_file"
