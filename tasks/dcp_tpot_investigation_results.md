# DCP TPOT Degradation Investigation Results

**Date:** 2026-03-14
**Hardware:** B200 Node 1 (colossus_b200_1), 8x B200 183GB
**Config:** DCP8 ag_rs, DeepSeek-V2-Lite
**Branch:** htphan/helix_a2a_rebased_main_fe294904c9

---

## Step 1: Quick Diagnostic Results (512K context)

| Test | Config Change | c=1 Median TPOT | c=2 Median TPOT | c=2 Mean TPOT | c=8 Median TPOT | c=8 Mean TPOT |
|------|--------------|:---------------:|:---------------:|:-------------:|:---------------:|:-------------:|
| 1A Baseline | none | 6.98 ms | 51.54 ms | 795.86 ms | 1228.73 ms | 2516.44 ms |
| 1B Mixed Chunk (chunk=32768) | `--enable-mixed-chunk` | 6.98 ms | **8.00 ms** | 177.04 ms | CRASHED | CRASHED |
| 1C No CUDA Graph | `--disable-cuda-graph` | 45.68 ms | 232.46 ms | 215.38 ms | 1323.89 ms | 2654.07 ms |
| 1D Sequential | 1 req at a time (×3) | 5.0 / 5.1 / 5.0 | — | — | — | — |

---

## Step 2: Mixed Chunk Tuning

### Step 2a: Smaller chunk sizes at 512K

| Test | chunk_size | c=1 Median | c=2 Median | c=2 Mean | c=8 |
|------|:----------:|:----------:|:----------:|:--------:|:---:|
| 1B (Step 1) | 32768 | 6.98 ms | 8.00 ms | 177.04 ms | CRASHED |
| 2A | 16384 | 7.01 ms | 7.61 ms | 14.68 ms | CRASHED |
| 2B | 8192 | 7.21 ms | 14.08 ms | 18.86 ms | CRASHED |

### Step 2b: Concurrency ceiling and other context lengths

| Test | Context | chunk_size | c=1 Median | c=2 Median | c=2 Mean | c=4 | c=8 |
|------|---------|:----------:|:----------:|:----------:|:--------:|:---:|:---:|
| 2E | 512K | 16384 | 7.01 ms | 7.61 ms | 14.68 ms | CRASHED | — |
| 2F | 256K | 16384 | 5.51 ms | CRASHED | — | CRASHED | CRASHED |
| 2G | 1M | 16384 | 7.13 ms | **8.25 ms** | 19.39 ms | — | — |

---

## Root Cause Analysis

### Why TP8 has good TPOT but DCP8 doesn't (both without mixed chunk)

**Both TP8 and DCP8 use the same scheduler with `enable_mixed_chunk=False`, chunked prefill, and prefill-first scheduling. Yet TP8 512K c=8 gets 37ms TPOT while DCP8 gets 2466ms.**

The root cause is **KV cache capacity asymmetry**:

| | TP8 | DCP8 |
|--|-----|------|
| `max_total_num_tokens` | 4,426,291 | 35,353,960 |
| Tokens per 512K request | 524,288 (full KV on each rank) | 65,536 (KV sharded across 8 ranks) |
| Max concurrent 512K reqs | ~8 | ~540 |
| Pool fills up at c= | **2** | **never** |

**DCP8's KV cache is 8x larger than TP8's** because DCP shards KV across ranks. Each rank only stores 1/8 of each request's KV cache.

**The scheduling consequence:**
- **TP8:** After 2 concurrent 512K requests, the token pool is ~24% full. The scheduler's `rem_total_tokens` check (in `PrefillAdder.budget_state()`) returns `NO_TOKEN` → `get_new_batch_prefill()` returns `None` → **decode runs**. Server logs show `gen throughput: 116 tok/s` with continuous decode batches.
- **DCP8:** Even with 5 concurrent 512K requests, the pool is only ~3% full. `get_new_batch_prefill()` **always** finds room for more prefill chunks → decode is **completely starved**. Server logs show `gen throughput: 1.5-2.5 tok/s` with decode batches 30-40s apart.

**Server log evidence:**
```
# TP8 512K: frequent decode, high throughput
Decode batch, #running-req: 2, token usage: 0.17, gen throughput: 116.11 tok/s
Decode batch, #running-req: 2, token usage: 0.17, gen throughput: 116.13 tok/s

# DCP8 512K: rare decode, almost zero throughput
Decode batch, #running-req: 2, token usage: 0.01, gen throughput: 1.98 tok/s
Decode batch, #running-req: 2, token usage: 0.02, gen throughput: 1.79 tok/s
```

### H1: Scheduler Decode Starvation from KV Overcapacity — CONFIRMED

The fix with `--enable-mixed-chunk` works because it merges decode tokens INTO the prefill batch, so decode happens even when prefill is scheduled. Results:
- 512K c=2: median TPOT drops from 51.5ms → **7.6ms** (with chunk=16384)
- 1M c=2: median TPOT drops to **8.25ms** (vs baseline 191ms mean)

### H3: CUDA Graph — Secondary Factor

Disabling CUDA graphs makes c=1 TPOT 6.5x worse (7ms → 46ms). CUDA graphs are critical for DCP decode perf but not the cause of the c>1 degradation.

### Mixed Chunk + DCP Watchdog Crashes

`--enable-mixed-chunk` crashes (300s watchdog) at higher concurrency because MIXED forward mode runs the **extend code path** with DCP NCCL overhead per layer (81+ calls). The crash correlates with KV capacity, not context length:
- 256K c=2: CRASHED (pool has 37.8M tokens → scheduler fits more concurrent requests)
- 512K c=2: OK, c=4: CRASHED
- 1M c=2: OK (pool has fewer tokens at mem_frac=0.65 → fewer concurrent requests)

---

## Conclusions

### The Real Problem

DCP's KV sharding creates a paradox: it enables much longer contexts (the whole point), but the 8x larger effective token pool causes the scheduler to **never** run out of prefill budget, starving decode completely.

### Recommended Fixes (in order of invasiveness)

1. **Quick fix — `--enable-mixed-chunk`:** Works for c<=2 at 512K/1M. Gives ~7-8ms TPOT. But crashes at higher concurrency.

2. **Better fix — Artificially cap `max_total_num_tokens` for DCP:** Set `max_total_num_tokens` to simulate TP-like memory pressure. E.g., for DCP8, divide by DCP size so the scheduler behaves like TP8. This would allow decode to interleave naturally without mixed chunk.

3. **Best fix — DCP-aware scheduling:** The scheduler should account for the fact that DCP KV is sharded. Either:
   - Report `max_total_num_tokens` in terms of **full-context tokens** (not per-rank tokens)
   - Add explicit `num_continuous_decode_steps` support (currently defined but not wired)
   - Implement a decode priority mechanism that guarantees decode runs after N prefill chunks

---

## Step 3: Input Length Scaling Verification

Tested with **same server config** (512K context, same KV pool), varying only input length at c=1,2,4,8.

| ISL | Chunks | c=1 Mean TPOT | c=2 Mean TPOT | c=4 Mean TPOT | c=8 Mean TPOT |
|-----|:------:|:-------------:|:-------------:|:-------------:|:-------------:|
| 64K | 2 | 4.34 ms | 10.80 ms | 11.07 ms | 14.38 ms |
| 128K | 4 | 4.70 ms | 10.89 ms | 15.98 ms | 23.82 ms |
| 256K | 8 | 5.24 ms | 37.23 ms | 103.96 ms | 64.11 ms |
| 512K | 16 | 7.00 ms | **787.93 ms** | **4222.17 ms** | **2497.01 ms** |

| ISL | Chunks | c=1 Median TPOT | c=2 Median TPOT | c=4 Median TPOT | c=8 Median TPOT |
|-----|:------:|:---------------:|:---------------:|:---------------:|:---------------:|
| 64K | 2 | 4.37 ms | 5.15 ms | 8.38 ms | 7.53 ms |
| 128K | 4 | 4.83 ms | 5.34 ms | 11.06 ms | 13.18 ms |
| 256K | 8 | 5.64 ms | 6.03 ms | 24.75 ms | 64.19 ms |
| 512K | 16 | 6.98 ms | 48.83 ms | 237.74 ms | **1218.13 ms** |

**Confirms scheduler hypothesis:** Degradation scales with input length (number of prefill chunks). The 256K→512K jump is nonlinear (64ms→2497ms at c=8, 39x worse for 2x more chunks), suggesting prefill duration crosses a critical threshold where decode is almost completely starved.

---

## Raw Log Files

B200 Node 1 container:
- `/output/investigation/` — Step 1 (1A-1D)
- `/output/investigation_step2/` — Step 2a (2A-2B)
- `/output/investigation_step2b/` — Step 2b (2E-2G)
- `/output/verify_scheduler/` — Step 3 (input length scaling)
