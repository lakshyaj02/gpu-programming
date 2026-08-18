#!/usr/bin/env bash

set -euo pipefail

week_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_root="$(dirname "$week_dir")"
build_dir="$week_dir/build"
output_file="$week_dir/results/gemm_results.csv"
iterations="${1:-20}"
warmup_iterations="${2:-5}"

if [[ ! "$iterations" =~ ^[1-9][0-9]*$ || ! "$warmup_iterations" =~ ^[0-9]+$ ]]; then
	printf 'Usage: %s [positive_iterations] [nonnegative_warmup_iterations]\n' "$0" >&2
	exit 1
fi

for command in cmake nvcc; do
	if ! command -v "$command" >/dev/null 2>&1; then
		printf 'error: required command not found: %s\n' "$command" >&2
		exit 1
	fi
done

cmake -S "$week_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" --target week05_tensor_cores -j
"$build_dir/week05_tensor_cores" "$iterations" "$warmup_iterations" > "$output_file"

printf 'Benchmark results written to %s\n' "$output_file"

python_command="python3"
if [[ -x "$project_root/.venv/bin/python" ]]; then
	python_command="$project_root/.venv/bin/python"
fi
if ! "$python_command" -c 'import matplotlib, numpy' >/dev/null 2>&1; then
	printf 'warning: matplotlib and numpy are unavailable; skipping plots\n' >&2
	exit 0
fi

"$python_command" "$week_dir/python/plot_results.py" --input "$output_file"