# Week 2: Memory Hierarchy and Tiling

This week focuses on how memory access patterns and shared-memory tiling affect
effective bandwidth and kernel runtime on GPU workloads.

## Environment

- GPU: GB200 NVL72
- CUDA version: 12.8
- Compiler: nvcc 12.4

## Build

From repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Week 2 layout

- `include/`: shared declarations for Week 2 kernels/utilities
- `src/`: CUDA kernels and benchmark drivers
- `python/`: optional plotting/comparison scripts
- `results/raw_out/`: raw benchmark CSV outputs
- `results/plots/`: generated comparison plots
- `results/observations.md`: final write-up

## Current results snapshot

Snapshot source: timestamp `20260728_035801` from `results/raw_out/`.

### Memory access (CUDA)

- Contiguous bandwidth (size=67108864, stride=1, iters=100): **2867.228 GB/s**
- Strided bandwidth (size=67108864, stride=64, iters=100): **447.559 GB/s**
- Relative drop (stride 64 vs contiguous): about **6.41x lower** bandwidth

### Matrix transpose (CUDA, disproportionate shape)

For `rows=3072`, `cols=777`, `iters=100`:

- `naive_b16`: `0.015197 ms`, `1256.524 GB/s`
- `tiled_t32_br8`: `0.009638 ms`, `1981.261 GB/s`
- Speedup: about **1.58x** in favor of tiled transpose

### Matrix multiplication (CUDA, ragged shape)

For `M=511`, `N=257`, `K=769`, `iters=100`:

- `naive_b16`: `0.109888 ms`, `1838.056 GFLOP/s`
- `tiled_t8`: `0.074265 ms`, `2719.733 GFLOP/s`
- Speedup: about **1.48x** in favor of tiled matmul

## Experiments

### 1) Memory access

Goal: measure cost of coalesced vs strided global memory reads/writes.

Implement:

- contiguous access kernel
- strided access kernels (for several strides)

Record:

- contiguous bandwidth (GB/s)
- strided bandwidth (GB/s)
- trend vs stride size

### 2) Matrix transpose

Goal: compare naive transpose with a coalesced/shared-memory transpose.

Implement:

- naive global-memory transpose
- tiled transpose using shared memory (with bank-conflict-safe padding)

Record:

- naive time (ms)
- coalesced/tiled time (ms)
- effective bandwidth (GB/s)
- speedup factor

### 3) Matrix multiplication

Goal: compare arithmetic intensity and memory behavior of naive vs tiled GEMM.

Implement:

- CPU reference matmul (correctness baseline)
- naive CUDA matmul
- tiled CUDA matmul using shared memory

Record:

- CPU reference time (ms)
- naive CUDA time (ms)
- tiled CUDA time (ms)
- speedup vs naive and vs CPU
- max absolute error (or relative error) vs CPU reference

## Reproducibility checklist

- Fix problem sizes and iteration count before comparisons
- Use warm-up launches before timing
- Report block size and tile size
- Synchronize before/after timed regions
- Store raw output in `results/raw_out/`
- Keep plots and write-up in `results/`

## Main observations template

Use this in `results/observations.md`:

1. Which access pattern most impacted bandwidth, and why?
2. How much did transpose coalescing improve throughput?
3. When did tiled GEMM outperform naive most strongly?
