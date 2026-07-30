# Week 3 Observations

## Reduction strategies benchmarked
- `atomic`: global atomicAdd per thread — baseline, bottlenecked by serialization
- `shared`: shared-memory tree reduction, one atomicAdd per block
- `warp`: warp-shuffle (`__shfl_down_sync`) reduction, one atomicAdd per warp-zero lane
- `hierarchical`: two-pass — per-block partial sums in phase 1, single-block final reduce in phase 2

## Results

<!-- Fill in after running benchmarks -->

## Main observations

<!-- Fill in after running benchmarks -->

## Q&A

- Why does atomicAdd on a single global address become a bottleneck?
- What is the advantage of `__shfl_down_sync` over shared-memory reads for intra-warp reduction?
- Why does the hierarchical kernel avoid a single global atomicAdd entirely?
