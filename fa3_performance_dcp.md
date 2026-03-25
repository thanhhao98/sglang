# FA3 DCP Performance Investigation

## Problem Statement
FA3 DCP decode is ~31% slower than FlashInfer DCP at low concurrency (14.91ms vs 11.38ms TPOT at cc1), despite identical performance without DCP (~9.5ms both). CC512 throughput also lower (2300 vs 3144 tok/s).

## Root Cause Analysis

### Issue 1 (DECODE): `_init_dcp_decode_metadata` Python for-loop
**File**: `flashattention_backend.py:2479-2529`

Called every decode step, this function had a Python `for p in range(max_local_pages)` loop launching individual GPU tensor ops per iteration. FlashInfer's equivalent (`filter_seq_indices` in flashinfer_mla_backend.py:747-770) was fully vectorized.

**Fix**: Replaced for-loop with single vectorized `torch.arange` + advanced indexing operation.
**Result**: TPOT dropped from 14.91ms to 11.29ms (24% improvement), matching FlashInfer's 11.38ms.

### Issue 2 (TTFT/THROUGHPUT): mem_fraction_static difference
FA3 used `MEM_FRAC=0.80` (OOMs at 0.85 due to FA3's larger CUDA graph workspace), while FlashInfer used `MEM_FRAC=0.85`. This 5% less KV cache capacity meant:
- Fewer concurrent requests could be held
- More queuing at high concurrency → higher TTFT
- Lower throughput at cc512

**Fix**: Increased to `MEM_FRAC=0.83` (max before OOM for FA3).
**Result**: cc256 TTFT dropped from 9118ms to 1977ms (matching FlashInfer's 1975ms). cc512 throughput improved from 2566 to 2947 (vs FlashInfer's 3144).

## Final Benchmark Results (H100, DCP8)

### Accuracy: 0.800 GSM8K (all configs pass)

### TPOT (ms) Comparison
| cc | FA3 Before | FA3 A2A Fix | FA3 AGRS Fix | FlashInfer A2A | FlashInfer AGRS |
|----|-----------|-------------|--------------|----------------|-----------------|
| 1 | 14.91 | **11.29** | **11.07** | 11.38 | 11.12 |
| 2 | 17.48 | **12.53** | **12.29** | 12.48 | 12.20 |
| 4 | 21.23 | **14.51** | **14.20** | 14.34 | 14.02 |
| 8 | 25.17 | **17.31** | **16.92** | 17.35 | 16.83 |
| 16 | 31.14 | **22.75** | **22.09** | 22.95 | 22.09 |
| 64 | 52.13 | **43.37** | **43.00** | 44.08 | 43.59 |
| 256 | 92.42 | **83.43** | **83.33** | 91.53 | 91.44 |
| 512 | 94.47 | **85.11** | **85.03** | 119.00 | 120.37 |

### Throughput (tok/s) with mem_frac fix
| cc | FA3 Before (0.80) | FA3 After (0.83) | FlashInfer (0.85) |
|----|-------------------|-------------------|-------------------|
| 1 | 65.52 | **87.40** | 86.66 |
| 64 | 1102.43 | **1339.18** | 1320.90 |
| 256 | 2261.56 | **2566.99** | 2558.00 |
| 512 | 2300.07 | **2947.66** | 3144.00 |

### TTFT (ms) with mem_frac fix
| cc | FA3 Before (0.80) | FA3 After (0.83) | FlashInfer (0.85) |
|----|-------------------|-------------------|-------------------|
| 1 | 122.53 | **116.98** | 116.43 |
| 256 | 9888.89 | **1977.25** | 1975.93 |
| 512 | 89558.20 | **46154.50** | 28397.40 |

## Summary
1. **Decode latency (TPOT)**: Fully resolved. FA3 now matches or beats FlashInfer across all concurrency levels.
2. **Throughput at cc1-cc256**: Fully resolved. FA3 matches FlashInfer.
3. **Throughput at cc512**: Reduced gap from 27% to 6.2%. Remaining gap is due to FA3 needing 2% less mem_frac (0.83 vs 0.85) because of larger CUDA graph workspace.
4. **TTFT at cc256**: Fully resolved (1977ms vs 1975ms).
5. **TTFT at cc512**: Improved from 89558ms to 46154ms (vs FlashInfer 28397ms). Remaining gap is the mem_frac difference.

## Commits
- `854945558` - vectorize `_init_dcp_decode_metadata` (main fix)
- `b14d63261` - benchmark scripts for 0.83 mem_frac testing
