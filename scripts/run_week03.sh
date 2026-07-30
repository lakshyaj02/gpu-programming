#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

skip_pytorch=false
ran_pytorch=false
raw_args=()
for arg in "$@"; do
	if [[ "$arg" == "--skip-pytorch" ]]; then
		skip_pytorch=true
		continue
	fi
	raw_args+=("$arg")
done

# Accept positional integers: [iterations] [warmup]
iterations_arg=""
warmup_arg=""
if [[ "${#raw_args[@]}" -ge 2 && "${raw_args[0]}" =~ ^[0-9]+$ && "${raw_args[1]}" =~ ^[0-9]+$ ]]; then
	iterations_arg="${raw_args[0]}"
	warmup_arg="${raw_args[1]}"
	raw_args=("${raw_args[@]:2}")
elif [[ "${#raw_args[@]}" -ge 1 && "${raw_args[0]}" =~ ^[0-9]+$ ]]; then
	iterations_arg="${raw_args[0]}"
	raw_args=("${raw_args[@]:1}")
fi

for command in cmake nvcc python3; do
	if ! command -v "$command" >/dev/null 2>&1; then
		printf 'error: required command not found: %s\n' "$command" >&2
		exit 1
	fi
done

results_dir="week03-reductions/results/raw"
mkdir -p "$results_dir"

timestamp="$(date +%Y%m%d_%H%M%S)"

printf '%s\n' 'Configuring and building week03_benchmark...'
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target week03_benchmark

cuda_csv="$results_dir/cuda_timings_reduction_${timestamp}.csv"

printf '%s\n' 'Running native CUDA benchmark harness...'
cuda_positional_args=()
[[ -n "$iterations_arg" ]] && cuda_positional_args+=("$iterations_arg")
[[ -n "$warmup_arg" ]] && cuda_positional_args+=("$warmup_arg")

./build/week03_benchmark "${cuda_positional_args[@]+"${cuda_positional_args[@]}"}" \
	"${raw_args[@]+"${raw_args[@]}"}" \
	> "$cuda_csv"

printf 'CUDA timings written to %s\n' "$cuda_csv"

if [[ "$skip_pytorch" == false ]]; then
	pytorch_passthrough_args=()
	[[ -n "$iterations_arg" ]] && pytorch_passthrough_args+=(--iterations "$iterations_arg")
	[[ -n "$warmup_arg" ]] && pytorch_passthrough_args+=(--warmup "$warmup_arg")

	if python3 -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
		printf '%s\n' 'Running PyTorch benchmark harness...'
		python3 week03-reductions/python/benchmark_pytorch.py \
			--timestamp "$timestamp" \
			--output-dir "$results_dir" \
			"${pytorch_passthrough_args[@]+"${pytorch_passthrough_args[@]}"}"
		ran_pytorch=true
	else
		printf '%s\n' 'warning: PyTorch CUDA not available; skipping PyTorch benchmark'
	fi
fi

printf 'Done. Week 3 benchmark artifacts written under %s\n' "$results_dir"
