# Week 07 observations

Run `./scripts/run_week07.sh` and record:

- Which operations benefit most from eliminating intermediate tensors?
- At what tensor size does launch overhead stop dominating?
- Where does fused softmax recomputation offset memory-traffic savings?
- How do row width and row count change occupancy and effective bandwidth?