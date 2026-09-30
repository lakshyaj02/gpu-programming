#!/usr/bin/env bash

set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
benchmark="${ATTENTION_BENCHMARK:-${root_dir}/build/week08_attention/week08_attention_benchmark}"

if [[ ! -x "${benchmark}" ]]; then
    echo "Benchmark not found: ${benchmark}" >&2
    exit 1
fi

command -v nsys >/dev/null || { echo "nsys is required" >&2; exit 1; }
command -v ncu >/dev/null || { echo "ncu is required" >&2; exit 1; }

nsys profile --force-overwrite true --output week08_attention "${benchmark}" 512 512 64 64 1
ncu --set basic --force-overwrite --export week08_attention "${benchmark}" 512 512 64 64 1
