# Week 05: Tensor Cores

Tensor Core precision and GEMM experiments using cuBLAS and CUDA WMMA.

## Experiments

- `precision_demo.cu` compares FP32, BF16, FP8 E4M3, and FP4 E2M1 representations.
- `wmma_gemm.cu` runs native TF32 and BF16 WMMA, both with FP32 accumulation.
- FP8 E4M3 and FP4 E2M1 inputs are quantized to their respective value sets and
	executed through FP16 WMMA. CSV rows label these paths `emulated_fp16_wmma` so
	their timings are not mistaken for native FP8 or FP4 Tensor Core performance.
- `main.cu` benchmarks both paths and verifies them against a CPU FP32 reference
	computed from the exact quantized inputs.

All matrices use row-major storage. Benchmark dimensions are multiples of 16, as
required by the WMMA kernel.

## Requirements

- CUDA Toolkit 12.8 or newer
- Blackwell NVIDIA GPU with compute capability 10.0 or 12.0
- CMake 3.24 or newer

Native FP8 requires Hopper or newer and native FP4 requires Blackwell with a
recent CUDA toolkit. Those formats use cuBLASLt or CUTLASS block-scaled APIs,
not the `<mma.h>` WMMA fragment API used in this exercise.

## Run

From this directory:

```bash
./scripts/run_benchmarks.sh [iterations] [warmup_iterations]
```

The script writes CSV output to `results/gemm_results.csv`. The precision demo is
written to standard error so it does not contaminate the CSV stream.

## Notes

A normal CUDA matrix-multiplication kernel usually assigns individual output emelents to individual threads. Each thread loads scalar values from $A$ and $B$, and executes scalar or vector fused multiply-add instructions on regular CUDA cores. A thread typically computes one or several elements, and each thread explicitly manages shared memory, synchronization, adn indexing.

WMMA instead describes a warp-level matrix operation:
```
wmma::wmma_syc(accumulator, fragmentA, fragmentB, accumulator);
```
All 32 threads in the warp cooperate to compute $$ D = A \times B + C $$
using tensor cores. One thread cannot perform a WMMA operation independently.

For BF16 `16 x 16 x 16` operation:
- `fragmentA` represents a `16 x 16` tile.
- `fragmentB` represents a `16 x 16` tile.
- `accumulator` represents a `16 x 16` output tile.
- The warp collectively calculates 256 output values.

Tensor cores process small matrix tiles directly and offer much greater GEMM throughput, particularly for low-precision inputs.

| Property | Normal CUDA Kernel | WMMA Kernel |
|---:|---:|---:|
| Execution Unit | Individual Thread | Entire Warp |
| Main instruction | Scalar/Vector FMA | Matrix multiply-accumulate |
| Hardware | CUDA Cores | Tensor Cores |
| Input Types | Broad Type Support | Specific supported Formats |
| Tile Structure | Programmer defined | Hardware-supported shapes |
| Value Ownership | Usually Explicit | Distributed and Opaque |
| Portability | Broad | Arcitecture dependent |

What Is a Fragment?
A fragment is a C++ object representing a matrix tile distributed across the registers of an entire warp:
```
wmma::fragment<
    wmma::matrix_a,
    16,
    16,
    16,
    __nv_bfloat16,
    wmma::row_major
> fragmentA;
```
Its template arguments describe:
```
role, M, N, K, element type, memory layout
```
A fragment is not a normal matrix container. Its values are distributed across the registers of the warp’s 32 threads. NVIDIA does not guarantee which thread owns which matrix element, and that mapping can differ by GPU architecture.

Every iteration loads one $A$ fragment and one $B$ fragment and accumulates their product into the same output fragment.

Also, “FP32 WMMA” in this project means FP32 values converted to TF32 precision for Tensor Core multiplication, generally with FP32 accumulation. It is not the same numerical operation as full IEEE FP32 multiplication on regular CUDA cores.


