- Why are tiny kernels inefficient?
```
At lower kernel sizes the kernel time does not scale well with work - hence it is inefficient. This is due to launch overhead and low occupancy, dominating useful competition.
```
- When does transfer time dominate?
```
Overhead ratio rises strongly at larger sizes in your current data, and block size 512 consistently shows the highest average ratio.
A few extreme max values indicate occasional timing outliers; median and p95 are more stable indicators than max.
Range of N: 4194304, 8388608, 16777216.
```
- Which block sizes perform similarly?
```
512 and 256 have very similar performance according to the latency distributions. 
```
- Why does block size stop mattering much for this kernel?
```
At a larger N, all reasonable block sizes converge because the kernel becomes memory-bandwidth limited rather than lauch/scheduling limited. Once memory throughput saturates, occupancy (degree of consurrency is warp scheduling) gives little benefit.
```
- Why is SAXPY usually bandwidth-bound?
```
SAXPY does very little math per byte moved. Arithmetic intensity is low: roughly 2 FLOPs for about 12 bytes (two reads, one write), so about 0.167 FLOP/byte. That pushes performance toward memory bandwidth limits.
```
- Why can PyTorch be close to custom CUDA for such a simple operation?
```
Both execute highly optimized elementwise GPU kernels with similar memory access patterns. For simple pointwise ops, there is little algorithmic advantage in hand-written code, so throughput can be similar.
```
- What is the difference between theoretical and effective bandwidth?
```
Theoretical bandwidth is the hardware peak from specs.
Effective bandwidth is measured from your run:
​BW(eff) = (3x4xN)/(time) GB/s for float32 vector add/SAXPY-style movement.
Effective is always below theoretical due to overheads, non-ideal access, and runtime effects.
```
- What happens when N is not divisible by the block size?
```
The final block is partially full, so some threads are inactive after the bounds check. Correctness is preserved by the idx < N guard, but utilization near the tail is lower, which can slightly reduce efficiency.
```