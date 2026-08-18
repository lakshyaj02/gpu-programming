#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
	printf 'Usage: %s [iterations] [warmup_iterations]\n' "$0"
	exit 0
fi

"$project_root/week05-tensor-cores/scripts/run_benchmarks.sh" "$@"