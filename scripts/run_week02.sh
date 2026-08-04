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

if [[ "${#raw_args[@]}" -ge 2 && "${raw_args[0]}" =~ ^[0-9]+$ && "${raw_args[1]}" =~ ^[0-9]+$ ]]; then
	raw_args=(--iterations "${raw_args[0]}" --warmup "${raw_args[1]}" "${raw_args[@]:2}")
elif [[ "${#raw_args[@]}" -ge 1 && "${raw_args[0]}" =~ ^[0-9]+$ ]]; then
	raw_args=(--iterations "${raw_args[0]}" "${raw_args[@]:1}")
fi

for command in cmake nvcc python3; do
	if ! command -v "$command" >/dev/null 2>&1; then
		printf 'error: required command not found: %s\n' "$command" >&2
		exit 1
	fi
done

results_dir="week02-memory-tiling/results/raw_out"
mkdir -p "$results_dir"

timestamp="$(date +%Y%m%d_%H%M%S)"

printf '%s\n' 'Configuring and building week02_benchmark...'
build_dir="week02-memory-tiling/build"
cmake -S week02-memory-tiling -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" -j --target week02_benchmark

printf '%s\n' 'Running native CUDA benchmark harness...'
python3 week02-memory-tiling/python/benchmark_cuda.py \
	--timestamp "$timestamp" \
	--output-dir "$results_dir" \
	"${raw_args[@]}"

if [[ "$skip_pytorch" == false ]]; then
	if python3 -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
		printf '%s\n' 'Running PyTorch benchmark harness...'
		python3 week02-memory-tiling/python/benchmark_pytorch.py \
			--timestamp "$timestamp" \
			--output-dir "$results_dir" \
			"${raw_args[@]}"
		ran_pytorch=true
	else
		printf '%s\n' 'warning: PyTorch CUDA not available; skipping PyTorch benchmark'
	fi
fi

if [[ "$ran_pytorch" == true ]]; then
	printf '%s\n' 'Generating merged CUDA/PyTorch comparison plots...'
	python3 week02-memory-tiling/python/merge_plot.py \
		--raw-dir "$results_dir" \
		--plots-dir "week02-memory-tiling/results/plots" \
		--timestamp "$timestamp"
fi

printf 'Done. Week 2 benchmark artifacts written under %s\n' "$results_dir"
