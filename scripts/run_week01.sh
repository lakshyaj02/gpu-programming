#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

for command in cmake nvcc python3; do
	if ! command -v "$command" >/dev/null 2>&1; then
		printf 'error: required command not found: %s\n' "$command" >&2
		exit 1
	fi
done

if [[ " $* " != *" --skip-pytorch "* ]]; then
	if ! python3 -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
		printf '%s\n' \
			'error: CUDA-enabled PyTorch is required (or pass --skip-pytorch)' >&2
		exit 1
	fi
fi

printf '%s\n' 'Building native CUDA benchmark...'
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

printf '%s\n' 'Running native CUDA and PyTorch benchmarks...'
python3 week01-vector-ops/python/benchmark_cuda.py "$@"
