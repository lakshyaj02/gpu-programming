# Week 4 Observations

## Performance Summary

Results for `M=N=K=4096` from the `20260808_022931` benchmark run. The naive and coarsened rows use their fastest measured tile size, which was tile 32 for both kernels.

| Kernel | Time | GFLOP/s | % of cuBLAS |
|--------|-----:|--------:|------------:|
| Naive | 18.155071 ms | 7,570.279 | 11.39% |
| Tiled 8 | 26.013617 ms | 5,283.347 | 7.95% |
| Tiled 16 | 17.535728 ms | 7,837.653 | 11.79% |
| Tiled 32 | 15.820498 ms | 8,687.397 | 13.07% |
| Coarsened | 9.877974 ms | 13,913.679 | 20.93% |
| cuBLAS | 2.067338 ms | 66,481.139 | 100.00% |

## Observations

- Why does the naive kernel repeatedly load the same values?

The naive kernel assigns one thread to every output element of the array $C$. Every thread independently computes $C_{i,j}$, where $$ C_{i,j} = \sum_{k=0}^{K-1} A_{ik}B_{ij} $$
Basically same rows of $A$ are repeatedly loaded for the same of $C$, and same columns of $B$ are repeatedly loaded for the same column of $C$. These values are fetched multiple times because threads cannot directly share register memory and there is no memory staged in the shared memory.

- Which matrix accesses are coalesced in a matrix multiplication?

Matrix $B$ reads are coalesced. At a fixed k, neighbouring threads read $$B[kN + j], B[kN + j + 1], B[kN + j + 2], ... $$
These are consecutive row major adressess, so the warp can combine them into a small number of memory transactions.
Matrix $C$ writes are coalesced. Neighboring threads write consecutive output columns. $$C[iN + j], C[iN + j + 1], C[iN + j + 2],... $$
Matrix $A$ reads are uniform broadcasts. Threads computing the same output row of $C$ all request $A[iK + k]. This is not a consecutive address pattern but the GPU can service it efficiently as a broadcast through cache hierarchy.

- How does shared-memory tiling increase data reuse?

Coperative loads into both shared-memory tiles are coalesced because threadIdx.x selects consecutive columns of row-major $A$ and $B$. The kernel then reuses these values when computing the output tile. For a tile size $T$:

- Each $A$ value loaded into shared memory is reused by $T$ threads computing different output columns.
- Each $B$ value is reused by $T$ threads computing different output rows.
- The block performs approximately $2T^3$ floating-point operations using only $2T^2$ global loads.

This reduces global memory traffic by roughky a factor of K, where without tiling a $T \times T$ takes $2T^2K$ global loads. Synchronization ensures complete threads are available before threads begin using them.

- Why must boundary tiles be filled with zeros?

This allows every thread to exucute with the same fixed size inner loop without divergent bounds checks for every multiplication. When the dimensions are not divisible by tile size, threads assigned to those positions cannot reads from global memory because doing so will lead to out of bounds accesses. Instead, those threads write $0.0f$ to their shared-memory tile positions.

- Why are two barriers required per tile?

This prevents race conditions between loading and reading shared-memory tiles.
1. Barrier after loading - Each thread loads a part of $A$ and $B$ ties. The first barrier ensures every load is complete before any thread starts reading the shared-memory values.
2. Barrier after computation - This ensures every thread has finished reading the current tiles before overwriting the tile with new data.

$$ load\ tile \rightarrow synchronize \rightarrow compute \rightarrow synchronize \rightarrow load\ next\ tile $$

- How many times is each loaded `A` value reused?

Each loaded $A$ value is reused $T$ times. $A_{ik}$ is used by $T$ threads computing different columns in the same tile row $$C_{i,j},C{i,j+1},...,C_{i,j+T-1}$$

- How many times is each loaded `B` value reused?

Each loaded $B$ value is reused $T$ times. $B_{kj}$ is used by $T$ threads computing different rows in the same tile column $$C_{i,j},C{i+1,j},...,C_{i+T-1,j}$$

- Why might a `32 × 32` block be slower than `16 × 16`?

1. Lower occupancy - A 1024-thread block may allow lesser resident blocks ber SM. A 256-thread block allows the scheduler to place several blocks on an SM as long as registers and shared memory permit.

2. Less scheduling flexibility - With only one resident block per SM, there are fewer independent warps available to hide memory, instruction and barrier latency.

3. Higher resource pressure - Since larger blocks consume more registers than shared memory per block.

4. More expensive synchronization - Synchronizing 1024 threads versus 256 threads.

However, tile 32 also provides more data reuse and arithmetic intensity. In these results it was faster for large matrices: tiled 32 reached 8,687 GFLOP/s, compared with 7,838 GFLOP/s for tiled 16 at 4096³. The potential occupancy cost was outweighed by better reuse on the tested GPU.

- How does thread coarsening improve register reuse?

Each thread computes multiple output elements instead of only one. In these results, coarsening improved the best tiled kernel from 8,687 GFLOP/s to 13,914 GFLOP/s at $4096^3$, a gain of about 60%.

- When does thread coarsening hurt occupancy?

The SM has a fixed register file. If each thread uses $R$ registers and a block contains $T$ threads, the block requires approximately: $$R_{block} = R \times T$$
Higher register usage can reduce the number of warps on an SM. If register demand is excessive, values may spill into local memory, which is backed by slow device memory.

Coarsening also reduces the number of threads used to cover an output tile. That is beneficial until there are too few active warps to hide memory and instruction latency. It particularly hurts when:

1. The coarsening factor us too high.
2. The kernel already has high register pressure.
3. Small matrices do not provide enough blocks to occupy all SMs.
4. Irregular boundaries leave some of each thread's outputs inactive.
5. Reduced occupancy is not offset by enough register and data reuse.

- Is the naive kernel memory-bound or compute-bound?

For 8 bytes of memory moved in the inner loop (4 bytes from $A$ and 4 bytes from $B$), the computation done is 2 FLOPs. The approximate arithemtic intensity is $$ AI_{native} \approx \frac{2\ FPLOs}{8\ bytes} = 0.25 FLOP/byte$$
This is too little computation per byte to approac the GPU's FP32 compute ceiling. Repeated $A$ broadcasts and cache hits can reduce actual DRAM traffic, but the kernel still issues redundant load instructions and has limited data reuse under explicit software control.

Thus, it is more precise to describe the naive kernel as memory-access and memory-latency bound, rather than purely DRAM-bandwidth bound.

- Does the tiled kernel move closer to the compute roof?

Yes. Shared-memory tiling increases arithmetic intensity by reusing each globally loaded value across multiple threads. For a $T\times T$ tile, the block performs approximately $2T^3$ FLOPs while loading $2T^2$ FP32 values, or $8T^2$ bytes. The arithmetic intensity is approximately : $$ AI_{tiles} \approx \frac{2T^3}{8T^2} = \frac{T}{4} FLOP/byte $$

| Tile size	| Approximate arithmetic intensity |
|---:|---|
| 8  |	2 FLOP/byte |
| 16 |	4 FLOP/byte |
| 32 |	8 FLOP/byte |

This is substantially higher than the naive kernel’s approximate 0.25 FLOP/byte, shifting the kernel rightward on a roofline plot and closer to the compute-bound region.

The measurements support this trend at $4096^3$:

Naive: 7,570 GFLOP/s
Tiled 16: 7,838 GFLOP/s
Tiled 32: 8,687 GFLOP/s

The coarsened kernel’s 13,914 GFLOP/s and cuBLAS’s 66,481 GFLOP/s show substantial compute capacity remains unused. Exact roofline placement requires measured DRAM traffic and the GPU’s FP32 peak.

- Why is cuBLAS still substantially faster?

Hierarchical tiling, Register blocking, Pipelining and double buffering, Architecture-specific instructions, Vectorized and asynchronous memory movement, Carefully tuned occupancy, Specialized kernel selection, Reduced synchronization and instruction overhead.

- Which optimization produced the largest improvement?

Thread coarsening produced the largest improvement among the custom-kernel optimizations at $4096^3$.
Relative to tiled 32, thread coarsening increased throughput by: $$ \frac{13,913.679−8,687.397}{8,867.397} \times 100 \approx 60.2\% $$
It also reduced execution time from 15.820 ms to 9.878 ms, a reduction of approximately 37.6%. The improvement comes from computing four outputs per thread, reusing operands across multiple register accumulators, and exposing more independent arithmetic work.

- What is the custom kernel’s performance as a percentage of cuBLAS?

The fastest custom kernel was the tile-32 thread-coarsened kernel:

Custom kernel: 13,913.679 GFLOP/s
cuBLAS: 66,481.139 GFLOP/s
Therefore: $$ \frac{13913}{66481} \times 100 \approx 21\%$$

The best custom kernel achieved 20.93% of cuBLAS throughput at $4096^3$. Equivalently, cuBLAS was approximately 4.78× faster.
