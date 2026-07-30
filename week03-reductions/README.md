# Week 3: Reductions

## Goal
Implement and benchmark parallel reduction strategies on the GPU, comparing atomic, shared-memory tree, warp-shuffle, and hierarchical approaches against a PyTorch baseline.

## Kernels

| Kernel | File | Strategy |
|--------|------|----------|
| `atomic` | `reduction_atomic.cu` | Each thread atomicAdd to a single global accumulator |
| `shared` | `reduction_shared.cu` | Tree reduction in shared memory, one atomicAdd per block |
| `warp` | `reduction_warp.cu` | Warp shuffle reduction (`__shfl_down_sync`), one atomicAdd per warp-zero lane |
| `hierarchical` | `reduction_hierarchical.cu` | Two-phase: phase 1 produces per-block partial sums; phase 2 reduces partials in a single block |

## Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target week03_benchmark
```

## Running

```bash
# CUDA benchmark (output CSV to stdout)
./build/week03_benchmark [iterations] [warmup_iterations]

# PyTorch baseline
python3 week03-reductions/python/benchmark_pytorch.py \
    --output-dir week03-reductions/results/raw
```

## Results
See [results/observations.md](results/observations.md).
