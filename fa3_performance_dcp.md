# FA3 DCP Performance Investigation

## Problem Statement
FA3 DCP decode is ~31% slower than FlashInfer DCP at low concurrency (14.91ms vs 11.38ms TPOT at cc1), despite identical performance without DCP (~9.5ms both).

## Root Cause Analysis

### Identified Bottleneck: `_init_dcp_decode_metadata` (flashattention_backend.py:2479-2529)

Called every decode step, this function has two performance issues:

1. **GPU-CPU sync** (line 2503): `local_seqlens.max().item()` forces CUDA synchronization
2. **Python for-loop** (lines 2512-2518): Iterates over `max_local_pages`, launching GPU tensor ops one-by-one from Python

FlashInfer's equivalent (`filter_seq_indices` in flashinfer_mla_backend.py:747-770) is fully vectorized with `torch.arange` + masking, no CPU sync.

### Why it converges at high concurrency
At cc256+, actual attention computation dominates and the per-step metadata overhead becomes a smaller fraction.

### TTFT gap (5x at cc256)
Cascading effect: slower decode steps cause request queuing, compounding latency.

## Fix Applied
- Vectorized the page table construction using torch tensor ops
- Removed `.max().item()` GPU-CPU sync by using upper bound from CUDA graph metadata or computing max on GPU
- Status: **IN PROGRESS**

## Benchmark Results
- TBD after deployment to colossus
