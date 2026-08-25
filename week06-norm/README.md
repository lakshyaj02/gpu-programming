## blockReduceSum() is a two level reduction:
1. Threads reduce their own warp.
2. One warp reduces the results of all warps.
For a block of 256 threads (8 warps $\times$ 32 threads)
```
Block: 256 threads

Warp 0: threads   0-31   ──reduce──> partial sum 0 ─┐
Warp 1: threads  32-63   ──reduce──> partial sum 1 ─┤
Warp 2: threads  64-95   ──reduce──> partial sum 2 ─┤
Warp 3: threads  96-127  ──reduce──> partial sum 3 ─┤
Warp 4: threads 128-159  ──reduce──> partial sum 4 ─┤
Warp 5: threads 160-191  ──reduce──> partial sum 5 ─┤
Warp 6: threads 192-223  ──reduce──> partial sum 6 ─┤
Warp 7: threads 224-255  ──reduce──> partial sum 7 ─┘
                                                       │
                                                       ▼
                                      Warp 0 reduces 8 partial sums
                                                       │
                                                       ▼
                                                Final block sum
                                        