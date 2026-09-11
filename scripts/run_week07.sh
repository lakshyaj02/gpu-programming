#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

iterations="${1:-100}"
warmup_iterations="${2:-10}"
if [[ ! "$iterations" =~ ^[1-9][0-9]*$ || ! "$warmup_iterations" =~ ^[0-9]+$ ]]; then
    printf 'Usage: %s [positive_iterations] [nonnegative_warmup_iterations]\n' "$0" >&2
    exit 1
fi

build_dir="build/week07-fused-kernel"
results_dir="week07-fused-kernel/results/raw"
plots_dir="week07-fused-kernel/results/plots"
mkdir -p "$results_dir" "$plots_dir"
timestamp="$(date +%Y%m%d_%H%M%S)"
csv="$results_dir/fusion_${timestamp}.csv"
plot="$plots_dir/fusion_speedup_${timestamp}.png"

cmake -S week07-fused-kernel -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" -j --target week07_fused_test week07_benchmark
"$build_dir/week07_fused_test"
"$build_dir/week07_benchmark" "$iterations" "$warmup_iterations" > "$csv"
python3 week07-fused-kernel/python/plot_results.py "$csv" --output "$plot"

printf 'Benchmark: %s\nPlot: %s\n' "$csv" "$plot"