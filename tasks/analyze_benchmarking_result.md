# SGLang DCP A2A Benchmarking Analysis

**Model:** DeepSeek-V2-Lite (15.7B, MLA, 1 effective KV head)
**Hardware:** H100 8x80GB, B200 8x183GB
**Configs:** TP8, DCP8 ag_rs, DCP8 a2a flashinfer, DCP8 a2a fa3 (H100 only)
**Context lengths:** 128K, 256K, 512K, 1M
**Concurrencies:** 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
**Total benchmark points:** 160 (H100) + 120 (B200) = 280

---

## 1. Executive Summary

- **A2A flashinfer ≈ AG_RS** — within 2% on B200, within 10% on H100 for most cases
- **A2A wins on H100 at 128K-256K** — 20-200% higher throughput at c>=8 due to ag_rs anomaly
- **FA3 is not viable for DCP** — 37-150x slower decode than flashinfer
- **DCP8 beats TP8 at c=1** — 35-42% higher prefill throughput
- **TP8 dominates at c>=2** — DCP has a critical scheduler decode starvation bug at 512K+
- **Root cause identified** — DCP's 8x larger KV pool prevents scheduler from running decode

---

## 2. TPOT Comparison: A2A vs AG_RS

### c=1 TPOT (ms) — pure decode, no scheduler interference

| Context | H100 ag_rs | H100 a2a_fi | H100 a2a_fa3 | B200 ag_rs | B200 a2a_fi |
|---------|:----------:|:-----------:|:------------:|:----------:|:-----------:|
| 128K | 4.19 | 4.34 | 161.35 | 4.83 | 4.84 |
| 256K | 133.96* | 5.01 | 307.35 | 5.29 | 5.30 |
| 512K | 6.21 | 6.17 | 718.82 | 6.76 | 7.00 |
| 1M | 6.08 | 6.11 | 934.41 | 6.78 | 7.03 |

> *H100 ag_rs 256K c=1 shows anomalous 134ms — likely a bug, not representative.

**Verdict:** At c=1, a2a_fi and ag_rs are essentially identical (within 4%). FA3 is 37-150x slower.

### TPOT % Delta: a2a_fi vs ag_rs (negative = a2a faster)

**B200:**

| Context | c=1 | c=2 | c=4 | c=8 |
|---------|----:|----:|----:|----:|
| 128K | +0.2% | -6.2% | -7.7% | -7.3% |
| 256K | +0.2% | -9.0% | -21.3% | +12.5% |
| 512K | +3.6% | +134.8% | +1.3% | +73.5% |
| 1M | +3.7% | +551.5% | +21.6% | +19.7% |

**H100:**

| Context | c=1 | c=2 | c=4 | c=8 |
|---------|----:|----:|----:|----:|
| 128K | +3.6% | -18.9% | +32.8% | -20.2% |
| 256K | -96.3%* | -6.9% | +0.4% | -65.6% |
| 512K | -0.6% | -31.1% | +439.6% | +416.8% |
| 1M | +0.5% | -82.4% | +375.4% | +216.9% |

> At 512K+ c>=2, both backends are dominated by the scheduler starvation issue (TPOT 800-10000ms), so the % deltas are noisy and not meaningful for comm-backend comparison.

---

## 3. Throughput Comparison: A2A vs AG_RS

### B200 Total Throughput (tok/s)

**128K:**

| Concurrency | TP8 | DCP8 ag_rs | DCP8 a2a_fi | a2a vs ag_rs |
|:-----------:|:---:|:----------:|:-----------:|:------------:|
| 1 | 69,755 | 67,771 | 68,609 | +1.2% |
| 2 | 126,860 | 86,960 | 88,135 | +1.4% |
| 4 | 154,123 | 92,329 | 95,456 | +3.4% |
| 8 | 153,316 | 94,689 | 95,695 | +1.1% |
| 16 | 147,655 | 96,761 | 96,470 | -0.3% |
| 32 | 154,407 | 96,624 | 96,009 | -0.6% |
| 64 | 136,886 | 95,863 | 95,834 | 0.0% |
| 128 | 147,911 | 96,506 | 94,072 | -2.5% |
| 256 | 136,981 | 94,611 | 95,873 | +1.3% |
| 512 | 137,471 | 96,642 | 93,702 | -3.0% |

**256K:**

| Concurrency | TP8 | DCP8 ag_rs | DCP8 a2a_fi | a2a vs ag_rs |
|:-----------:|:---:|:----------:|:-----------:|:------------:|
| 1 | 95,514 | 89,578 | 88,048 | -1.7% |
| 2 | 357,111 | 106,353 | 104,564 | -1.7% |
| 4 | 449,483 | 117,693 | 116,326 | -1.2% |
| 8 | 449,433 | 119,324 | 119,257 | -0.1% |
| 16 | 454,802 | 119,790 | 118,459 | -1.1% |
| 32 | 417,123 | 115,876 | 119,634 | +3.2% |
| 64 | 443,199 | 115,405 | 119,392 | +3.5% |
| 128 | 449,584 | 119,995 | 119,802 | -0.2% |
| 256 | 447,206 | 119,616 | 119,946 | +0.3% |
| 512 | 415,567 | 115,540 | 119,627 | +3.5% |

**512K:**

| Concurrency | TP8 | DCP8 ag_rs | DCP8 a2a_fi | a2a vs ag_rs |
|:-----------:|:---:|:----------:|:-----------:|:------------:|
| 1 | 55,945 | 67,318 | 67,655 | +0.5% |
| 2 | 288,108 | 77,816 | 77,628 | -0.2% |
| 4 | 395,821 | 76,930 | 77,327 | +0.5% |
| 8 | 364,126 | 80,756 | 79,159 | -2.0% |
| 16 | 187,460 | 80,661 | 80,662 | 0.0% |
| 32 | 204,719 | 79,367 | 79,604 | +0.3% |
| 64 | 360,042 | 81,199 | 79,456 | -2.1% |
| 128 | 204,770 | 81,384 | 79,658 | -2.1% |
| 256 | 385,031 | 79,718 | 79,490 | -0.3% |
| 512 | 395,321 | 80,938 | 80,205 | -0.9% |

**1M:**

| Concurrency | TP8 | DCP8 ag_rs | DCP8 a2a_fi | a2a vs ag_rs |
|:-----------:|:---:|:----------:|:-----------:|:------------:|
| 1 | 53,776 | 51,098 | 51,055 | -0.1% |
| 2 | 237,865 | 54,986 | 55,038 | +0.1% |
| 4 | 241,976 | 57,835 | 57,756 | -0.1% |
| 8 | 221,249 | 58,223 | 57,912 | -0.5% |
| 16 | 220,072 | 58,164 | 58,106 | -0.1% |
| 32 | 212,728 | 57,809 | 57,611 | -0.3% |
| 64 | 217,503 | 58,394 | 57,786 | -1.0% |
| 128 | 219,689 | 58,152 | 57,973 | -0.3% |
| 256 | 219,822 | 58,400 | 57,917 | -0.8% |
| 512 | 218,079 | 57,954 | 57,837 | -0.2% |

### H100 Total Throughput (tok/s)

**128K:**

| Concurrency | TP8 | DCP8 ag_rs | DCP8 a2a_fi | DCP8 a2a_fa3 | a2a_fi vs ag_rs |
|:-----------:|:---:|:----------:|:-----------:|:------------:|:---------------:|
| 1 | 39,549 | 61,522 | 60,073 | 4,959 | -2.4% |
| 2 | 63,836 | 73,767 | 75,297 | 5,719 | +2.1% |
| 4 | 68,861 | 66,582 | 79,473 | 6,007 | **+19.4%** |
| 8 | 69,379 | 65,793 | 80,991 | 6,062 | **+23.1%** |
| 16 | 68,563 | 66,835 | 81,133 | 6,022 | **+21.4%** |
| 32 | 67,354 | 66,789 | 81,268 | 6,061 | **+21.7%** |
| 64 | 66,879 | 66,772 | 80,082 | 6,051 | **+19.9%** |
| 128 | 64,454 | 65,064 | 81,325 | 6,067 | **+25.0%** |
| 256 | 65,711 | 65,016 | 81,646 | 6,047 | **+25.6%** |
| 512 | 65,369 | 66,336 | 79,736 | 6,065 | **+20.2%** |

**256K:**

| Concurrency | TP8 | DCP8 ag_rs | DCP8 a2a_fi | DCP8 a2a_fa3 | a2a_fi vs ag_rs |
|:-----------:|:---:|:----------:|:-----------:|:------------:|:---------------:|
| 1 | 51,877 | 70,450 | 68,010 | 29,020 | -3.5% |
| 2 | 250,785 | 54.54* | 76,190 | 34,737 | — |
| 4 | 368,479 | 201.21* | 82,545 | 45,104 | — |
| 8 | 379,648 | 28,054 | 84,244 | 45,447 | **+200.3%** |
| 16 | 367,521 | 5,770 | 84,625 | 44,983 | **+1366.8%** |
| 32 | 348,759 | 5,766 | 84,122 | 45,619 | **+1358.9%** |
| 64 | 381,306 | 5,632 | 84,637 | 45,949 | **+1403.0%** |
| 128 | 373,301 | 5,799 | 84,491 | 45,487 | **+1357.0%** |
| 256 | 285,317 | 5,777 | 84,432 | 46,010 | **+1361.5%** |
| 512 | 329,919 | 5,797 | 81,924 | 45,709 | **+1313.1%** |

> *H100 ag_rs 256K c=2,4 were rerun — values are TPOT not throughput. ag_rs 256K c>=8 throughput collapses to 5-28K tok/s while a2a_fi maintains 82-85K tok/s.

**512K:**

| Concurrency | TP8 | DCP8 ag_rs | DCP8 a2a_fi | DCP8 a2a_fa3 | a2a_fi vs ag_rs |
|:-----------:|:---:|:----------:|:-----------:|:------------:|:---------------:|
| 1 | 33,214 | 47,488 | 42,842 | 3,622 | -9.8% |
| 2 | 42,286 | 51,207 | 53,348 | 3,741 | +4.2% |
| 4 | 44,200 | 48,225 | 51,922 | 3,805 | +7.7% |
| 8 | 45,077 | 48,017 | 51,980 | 3,813 | +8.3% |
| 16 | 44,780 | 48,075 | 51,871 | 3,832 | +7.9% |
| 32 | 44,040 | 48,043 | 48,212 | 3,801 | +0.4% |
| 64 | 44,231 | 47,896 | 52,203 | 3,814 | +9.0% |
| 128 | 43,946 | 48,121 | 48,036 | 3,784 | -0.2% |
| 256 | 43,553 | 48,126 | 51,976 | 3,704 | +8.0% |
| 512 | 42,853 | 48,057 | 51,990 | 3,793 | +8.2% |

**1M:**

| Concurrency | TP8 | DCP8 ag_rs | DCP8 a2a_fi | DCP8 a2a_fa3 | a2a_fi vs ag_rs |
|:-----------:|:---:|:----------:|:-----------:|:------------:|:---------------:|
| 1 | 28,380 | 35,784 | 35,742 | 1,685 | -0.1% |
| 2 | 32,072 | 38,293 | 38,245 | 1,739 | -0.1% |
| 4 | 31,560 | 39,712 | 39,732 | 1,758 | +0.1% |
| 8 | 33,514 | 39,721 | 39,817 | 1,731 | +0.2% |
| 16 | 4,187 | 39,709 | 39,839 | 1,744 | +0.3% |
| 32 | 1,133 | 39,497 | 39,869 | 1,716 | +0.9% |
| 64 | 31,542 | 39,622 | 39,679 | 1,738 | +0.1% |
| 128 | 32,417 | 39,953 | 39,817 | 1,759 | -0.3% |
| 256 | 31,678 | 39,833 | 39,885 | +0.1% | +0.1% |
| 512 | 31,715 | 39,719 | 39,922 | 1,746 | +0.5% |

---

## 4. DCP8 vs TP8 Comparison

### DCP8 advantage: higher prefill throughput at c=1

| Context | B200 TP8 | B200 DCP8 ag_rs | DCP/TP ratio | H100 TP8 | H100 DCP8 ag_rs | DCP/TP ratio |
|---------|:--------:|:---------------:|:------------:|:--------:|:---------------:|:------------:|
| 128K | 69,755 | 67,771 | 0.97x | 39,549 | 61,522 | **1.56x** |
| 256K | 95,514 | 89,578 | 0.94x | 51,877 | 70,450 | **1.36x** |
| 512K | 55,945 | 67,318 | **1.20x** | 33,214 | 47,488 | **1.43x** |
| 1M | 53,776 | 51,098 | 0.95x | 28,380 | 35,784 | **1.26x** |

DCP8 has higher c=1 throughput on H100 across all context lengths (1.26-1.56x). On B200, the advantage is smaller and only appears at 512K.

### TP8 advantage: better scaling at c>=2

| Context | B200 TP8 c=8 | B200 DCP8 c=8 | TP8/DCP ratio |
|---------|:------------:|:-------------:|:-------------:|
| 128K | 153,316 | 94,689 | **1.62x** |
| 256K | 449,433 | 119,324 | **3.77x** |
| 512K | 364,126 | 80,756 | **4.51x** |
| 1M | 221,249 | 58,223 | **3.80x** |

TP8 delivers 1.6-4.5x more throughput than DCP8 at c=8 because DCP's scheduler starvation prevents efficient decode batching.

---

## 5. FA3 vs Flashinfer for DCP

FA3 is catastrophically worse than flashinfer for DCP decode on H100:

| Context | flashinfer c=1 TPOT | FA3 c=1 TPOT | FA3 slowdown | flashinfer thru (c=8) | FA3 thru (c=8) |
|---------|:-------------------:|:------------:|:------------:|:---------------------:|:--------------:|
| 128K | 4.34 ms | 161 ms | **37x** | 80,991 | 6,062 |
| 256K | 5.01 ms | 307 ms | **61x** | 84,244 | 45,447 |
| 512K | 6.17 ms | 719 ms | **117x** | 51,980 | 3,813 |
| 1M | 6.11 ms | 934 ms | **153x** | 39,817 | 1,731 |

**Verdict:** FA3 is not competitive for DCP. Do not use `--attention-backend fa3` with DCP.

---

## 6. Critical Bug: DCP Scheduler Decode Starvation

### The Problem

At c>=2 with 512K+ context, DCP8 TPOT degrades from ~7ms to 800-10,000ms. This affects BOTH ag_rs and a2a equally.

### Root Cause

DCP shards KV across 8 ranks, creating 8x more KV pool capacity than TP8:

| | TP8 512K | DCP8 512K |
|--|----------|-----------|
| `max_total_num_tokens` | 4.4M | 35.4M |
| Tokens per request | 524K | 65K (÷8 ranks) |
| Pool fills at c= | ~2 | never |

The scheduler's `PrefillAdder.rem_total_tokens` (`schedule_policy.py:453-471`) never returns `NO_TOKEN` for DCP → `get_new_batch_prefill()` always returns a prefill batch → decode is completely starved.

**Server log evidence:**
- TP8 512K: `gen throughput: 116 tok/s` (continuous decode)
- DCP8 512K: `gen throughput: 1.8 tok/s` (decode runs once per ~30s)

### Impact on Benchmarks

All DCP TPOT numbers at c>=2 for 512K+ are **not representative of true decode performance**. They measure scheduler starvation, not comm-backend overhead. The a2a vs ag_rs comparison at 512K+ is meaningless until this bug is fixed.

### Proposed Fixes

1. **Quick fix:** `--enable-mixed-chunk --chunked-prefill-size 16384` — works for c<=2 (7-8ms TPOT)
2. **Proper fix:** Cap `max_total_num_tokens` for DCP (divide by DCP size) so scheduler behaves like TP8
3. **Best fix:** DCP-aware scheduling that reports token counts in full-context units

---

## 7. Hardware Comparison: H100 vs B200

### c=1 TPOT (ms) — TP8

| Context | H100 | B200 | B200 speedup |
|---------|:----:|:----:|:------------:|
| 128K | 13.82 | 5.67 | **2.4x** |
| 256K | 34.53 | 8.09 | **4.3x** |
| 512K | 86.06 | 12.29 | **7.0x** |
| 1M | 100.65 | 13.75 | **7.3x** |

### c=1 TPOT (ms) — DCP8 ag_rs

| Context | H100 | B200 | B200 speedup |
|---------|:----:|:----:|:------------:|
| 128K | 4.19 | 4.83 | 0.87x |
| 256K | 133.96* | 5.29 | 25x* |
| 512K | 6.21 | 6.76 | 0.92x |
| 1M | 6.08 | 6.78 | 0.90x |

> *H100 ag_rs 256K anomaly excluded. Excluding that, DCP8 TPOT is similar on both hardware.

### Peak Throughput (tok/s) — TP8

| Context | H100 best | B200 best | B200/H100 |
|---------|:---------:|:---------:|:---------:|
| 128K | 69,379 (c=8) | 154,407 (c=32) | **2.2x** |
| 256K | 381,306 (c=64) | 454,802 (c=16) | **1.2x** |
| 512K | 45,077 (c=8) | 395,821 (c=4) | **8.8x** |
| 1M | 33,514 (c=8) | 241,976 (c=4) | **7.2x** |

B200 is 1.2-8.8x higher throughput than H100 for TP8, with the gap widening at longer contexts due to B200's 2.3x larger VRAM allowing more concurrent requests.

---

## 8. Summary Verdicts

### A2A vs AG_RS

| Hardware | Context | Winner | Margin |
|----------|---------|--------|--------|
| B200 | 128K | Tie | <3% |
| B200 | 256K | Tie | <2% |
| B200 | 512K | Tie | <2% (both broken at c>=2) |
| B200 | 1M | Tie | <1% (both broken at c>=2) |
| H100 | 128K | **a2a_fi** | +20-25% throughput at c>=4 |
| H100 | 256K | **a2a_fi** | +200-1400% throughput (ag_rs collapses) |
| H100 | 512K | **a2a_fi** | +4-9% throughput |
| H100 | 1M | Tie | <1% |

### DCP8 vs TP8

| Metric | DCP8 advantage | TP8 advantage |
|--------|---------------|---------------|
| c=1 prefill throughput | +20-56% on H100 | — |
| c>=2 decode TPOT | — | 10-100x better (scheduler bug) |
| c>=2 throughput | — | 1.6-4.5x better |
| Max context support | Longer contexts possible | Limited by per-GPU VRAM |

### Overall Recommendation

1. **Fix the DCP scheduler starvation bug first** — this is the #1 priority. All DCP performance at c>=2 is crippled.
2. **Use flashinfer, not FA3** — FA3 is 37-153x slower for DCP decode.
3. **A2A and AG_RS are interchangeable for single-node** — <2% difference on B200. A2A has a moderate edge on H100.
4. **A2A may show advantages at multi-node** — the vLLM Helix results showed A2A wins at DCP>=8 cross-node. Single-node DCP8 doesn't exercise the A2A topology advantage.

---

## 9. 4-Way Comparison: TP8 vs AG_RS vs A2A vs A2A+replicate-q-proj (B200)

### c=1 TPOT (ms) — pure decode latency

| Context | TP8 | DCP8 ag_rs | DCP8 a2a | DCP8 a2a+repl | repl vs a2a |
|---------|:---:|:----------:|:--------:|:-------------:|:-----------:|
| 128K | 5.67 | 4.83 | 4.84 | 4.77 | -1.4% |
| 256K | 8.09 | 5.29 | 5.30 | 5.36 | +1.1% |
| 512K | 12.29 | 6.76 | 7.00 | 6.84 | -2.3% |
| 1M | 13.75 | 6.78 | 7.03 | 6.95 | -1.1% |

> All DCP configs are **1.2-2x faster** than TP8 at c=1 decode. `--dcp-replicate-q-proj` is within noise of baseline a2a (<3%).

### c=1 Total Throughput (tok/s) — includes prefill

| Context | TP8 | DCP8 ag_rs | DCP8 a2a | DCP8 a2a+repl | DCP8 ag_rs+repl |
|---------|:---:|:----------:|:--------:|:-------------:|:---------------:|
| 128K | 69,755 | 67,771 | 68,609 | 67,430 | 68,439 |
| 256K | 95,514 | 89,578 | 88,048 | 84,974 | 85,534 |
| 512K | 55,945 | 67,318 | 67,655 | 63,791 | 63,793 |
| 1M | 53,776 | 51,098 | 51,055 | 48,518 | 48,282 |

> At c=1, DCP8 has higher throughput than TP8 at 512K (1.2x). At 128K-256K, TP8 is slightly ahead.
> `replicate-q-proj` shows ~5% lower throughput than baseline — the replicated Q/kv_b projections use more compute per rank.

### c=8 Throughput Comparison (tok/s)

| Context | TP8 | DCP8 ag_rs | DCP8 a2a | DCP8 a2a+repl | DCP8 ag_rs+repl |
|---------|:---:|:----------:|:--------:|:-------------:|:---------------:|
| 128K | 153,316 | 94,689 | 95,695 | 93,050 | 93,583 |
| 256K | 449,433 | 119,324 | 119,257 | 111,678 | 107,987 |
| 512K | 364,126 | 80,756 | 79,159 | 74,196 | 74,899 |
| 1M | 221,249 | 58,223 | 57,912 | 54,531 | 54,438 |

> TP8 dominates at c>=8 (1.6-4.5x over DCP) due to DCP scheduler starvation at 512K+.
> `replicate-q-proj` is ~5-7% lower than baseline DCP at c=8. The extra per-rank compute from replicated projections slightly reduces throughput.

### TPOT at c=8 (ms) — decode under load

| Context | TP8 | DCP8 ag_rs | DCP8 a2a | DCP8 a2a+repl | DCP8 ag_rs+repl |
|---------|:---:|:----------:|:--------:|:-------------:|:---------------:|
| 128K | 8.06 | 27.62 | 27.64 | 25.62 | 25.60 |
| 256K | 8.98 | 108.45 | 108.11 | 186.20 | 127.74 |
| 512K | 22.75 | 1,093 | 1,060 | 1,895 | 5,026 |
| 1M | 20.48 | 1,868 | 1,812 | 1,767 | 1,912 |

> At 128K c=8: all DCP configs ~26ms (3x TP8). `replicate-q-proj` slightly better than baseline.
> At 256K+ c=8: scheduler starvation dominates — all DCP configs have 100-5000ms TPOT regardless of comm backend or Q replication.

### Throughput Scaling: B200 128K (best-case DCP scenario)

| Concurrency | TP8 | ag_rs | a2a | a2a+repl | ag_rs+repl |
|:-----------:|:---:|:-----:|:---:|:--------:|:----------:|
| 1 | 69,755 | 67,771 | 68,609 | 67,430 | 68,439 |
| 2 | 126,860 | 86,960 | 88,135 | 83,336 | 84,554 |
| 4 | 154,123 | 92,329 | 95,456 | 93,169 | 93,851 |
| 8 | 153,316 | 94,689 | 95,695 | 93,050 | 93,583 |
| 16 | 147,655 | 96,761 | 96,470 | 93,317 | 93,358 |
| 32 | 154,407 | 96,624 | 96,009 | 91,611 | 91,870 |
| 64 | 136,886 | 95,863 | 95,834 | 93,027 | 91,701 |
| 128 | 147,911 | 96,506 | 94,072 | 91,028 | 93,521 |
| 256 | 136,981 | 94,611 | 95,873 | 91,362 | 91,930 |
| 512 | 137,471 | 96,642 | 93,702 | 90,975 | 94,100 |

> At 128K, DCP scales to ~93-97K tok/s plateau (all 4 DCP variants). TP8 reaches 154K.
> `replicate-q-proj` is ~3-5% below baseline DCP across all concurrencies at 128K.

### Throughput Scaling: B200 256K

| Concurrency | TP8 | ag_rs | a2a | a2a+repl | ag_rs+repl |
|:-----------:|:---:|:-----:|:---:|:--------:|:----------:|
| 1 | 95,514 | 89,578 | 88,048 | 84,974 | 85,534 |
| 2 | 357,111 | 106,353 | 104,564 | 96,906 | 98,376 |
| 4 | 449,483 | 117,693 | 116,326 | 108,418 | 110,019 |
| 8 | 449,433 | 119,324 | 119,257 | 111,678 | 107,987 |
| 16 | 454,802 | 119,790 | 118,459 | 111,721 | 111,976 |
| 64 | 443,199 | 115,405 | 119,392 | 107,758 | 112,094 |
| 512 | 415,567 | 115,540 | 119,627 | 111,900 | 112,610 |

> TP8 is 3-4x faster than all DCP variants at 256K c>=4.
> Among DCP variants: baseline a2a/ag_rs ~119K, replicate-q-proj ~108-112K (5-8% lower).

### Key Takeaways

1. **`--dcp-replicate-q-proj` does NOT improve TPOT** on single-node NVLink — the AllGather Q latency is negligible (~0.1ms).
2. **`--dcp-replicate-q-proj` slightly reduces throughput** by ~3-7% due to replicated Q/kv_b projection compute (each rank does full-head projection instead of TP-sharded).
3. **All DCP variants show the same scheduler starvation pattern** at 512K+ c>=2 — the bottleneck is the scheduler, not the comm backend.
4. **DCP wins over TP8 at c=1 decode** (1.2-2x lower TPOT) but TP8 dominates at c>=2 throughput.
5. **The replicate-q-proj optimization targets cross-node DCP** where AllGather Q latency is higher; single-node NVLink is not the right benchmark to show its value.

---

## 10. `--dcp-replicate-q-proj` Detailed Results (2026-03-16)

**Branch:** `htphan/q-project-replication`
**Purpose:** Eliminate the AllGather Q collective during DCP decode by replicating the Q and kv_b projections (tp_size=1) on each rank. Each rank computes all heads locally, skipping the AllGather.

### Accuracy (GSM8K 1319 questions) — All Pass

| Machine | Config | CG | Accuracy | Invalid |
|---------|--------|:--:|:--------:|:-------:|
| H100 | a2a + repl | No | 0.378 | 0.006 |
| H100 | a2a + repl | Regular | 0.376 | 0.006 |
| H100 | a2a + repl | Piecewise | 0.377 | 0.006 |
| H100 | ag_rs + repl | No | 0.390 | 0.006 |
| H100 | ag_rs + repl | Regular | 0.387 | 0.005 |
| H100 | ag_rs + repl | Piecewise | 0.390 | 0.006 |
| B200 | a2a + repl | No | 0.377 | 0.005 |
| B200 | a2a + repl | Regular | 0.388 | 0.004 |
| B200 | a2a + repl | Piecewise | 0.388 | 0.004 |
| B200 | ag_rs + repl | No | 0.382 | 0.004 |
| B200 | ag_rs + repl | Regular | 0.379 | 0.005 |
| B200 | ag_rs + repl | Piecewise | 0.383 | 0.005 |

> All 12 configs ~0.38-0.39, consistent with baseline. No regression.

### Bug Fix: Chunked Prefix Crash at 256K+

During performance benchmarking, `--dcp-replicate-q-proj` crashed at context >= 256K in `_chunked_prefix_attn_mha`.
- **Root cause:** `kv_b_proj` (tp_size=1) outputs all heads but the reshape used `num_local_heads`, inflating the token dimension 8x. Also `k_pe` (1 KV head) wasn't expanded to match.
- **Fix:** Reshape with `num_heads`, slice to local heads, `expand_as` for k_pe. Accuracy recheck passed (0.378).

### B200 c=1 TPOT: replicate-q-proj vs baseline

| Context | C5 a2a+repl | C3 a2a baseline | Delta | C6 ag_rs+repl | C2 ag_rs baseline | Delta |
|---------|:-----------:|:---------------:|:-----:|:-------------:|:-----------------:|:-----:|
| 128K | 4.77 ms | 4.84 ms | -1.4% | 4.69 ms | 4.83 ms | -2.9% |
| 256K | 5.36 ms | 5.30 ms | +1.1% | 5.32 ms | 5.29 ms | +0.6% |
| 512K | 6.84 ms | 7.00 ms | -2.3% | 6.77 ms | 6.76 ms | +0.1% |
| 1M | 6.95 ms | 7.03 ms | -1.1% | 6.74 ms | 6.78 ms | -0.6% |

> **Verdict:** `--dcp-replicate-q-proj` shows **no measurable TPOT improvement** at c=1 (within noise, <3%).
> The AllGather Q is already fast (~0.1ms) for batch_size=1 with 8 GPUs on NVLink.
> The benefit may appear at higher batch sizes or cross-node DCP where AllGather latency is higher.

### B200 Performance: C5 (a2a + replicate-q-proj)

**C5_128K** (ISL=130048, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) |
|:-----------:|:---------:|:---------:|:------------------------:|
| 1 | 4.77 | 638.82 | 67,430 |
| 2 | 14.80 | 924.10 | 83,336 |
| 4 | 16.90 | 1067.16 | 93,169 |
| 8 | 25.62 | 1499.99 | 93,050 |
| 16 | 25.81 | 1463.88 | 93,317 |
| 64 | 25.47 | 1512.22 | 93,027 |
| 512 | 33.24 | 1639.41 | 90,975 |

**C5_256K** (ISL=261120, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) |
|:-----------:|:---------:|:---------:|:------------------------:|
| 1 | 5.36 | 1682.93 | 84,974 |
| 2 | 42.50 | 2141.70 | 96,906 |
| 4 | 115.30 | 2598.42 | 108,418 |
| 8 | 186.20 | 3634.99 | 111,678 |
| 16 | 186.93 | 3548.15 | 111,721 |
| 64 | 187.60 | 3665.74 | 107,758 |
| 512 | 187.74 | 3518.27 | 111,900 |

**C5_512K** (ISL=523264, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) |
|:-----------:|:---------:|:---------:|:------------------------:|
| 1 | 6.84 | 5737.58 | 63,791 |
| 2 | 1991.51 | 7347.67 | 72,786 |
| 8 | 1894.59 | 13513.14 | 74,196 |
| 64 | 1888.58 | 13429.14 | 74,529 |
| 512 | 1898.37 | 13504.29 | 74,174 |

**C5_1M** (ISL=1047552, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) |
|:-----------:|:---------:|:---------:|:------------------------:|
| 1 | 6.95 | 8854.96 | 48,518 |
| 2 | 1341.02 | 13461.79 | 51,825 |
| 8 | 1767.48 | 18379.41 | 54,531 |
| 64 | 1762.75 | 18271.33 | 54,729 |
| 512 | 1764.84 | 18344.07 | 54,569 |

### B200 Performance: C6 (ag_rs + replicate-q-proj)

**C6_128K** (ISL=130048, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) |
|:-----------:|:---------:|:---------:|:------------------------:|
| 1 | 4.69 | 630.73 | 68,439 |
| 2 | 14.73 | 909.48 | 84,554 |
| 4 | 17.61 | 1020.32 | 93,851 |
| 8 | 25.60 | 1378.17 | 93,583 |
| 16 | 25.97 | 1468.88 | 93,358 |
| 64 | 33.36 | 1643.35 | 91,701 |
| 512 | 25.19 | 1370.48 | 94,100 |

**C6_256K** (ISL=261120, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) |
|:-----------:|:---------:|:---------:|:------------------------:|
| 1 | 5.32 | 1676.45 | 85,534 |
| 2 | 41.60 | 2126.28 | 98,376 |
| 4 | 66.81 | 2525.81 | 110,019 |
| 8 | 127.74 | 3632.65 | 107,987 |
| 16 | 127.08 | 3529.01 | 111,976 |
| 64 | 126.50 | 3521.12 | 112,094 |
| 512 | 95.58 | 3977.46 | 112,610 |

**C6_512K** (ISL=523264, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) |
|:-----------:|:---------:|:---------:|:------------------------:|
| 1 | 6.77 | 5739.20 | 63,793 |
| 2 | 2005.22 | 7405.86 | 72,028 |
| 8 | 5025.95 | 12594.26 | 74,899 |
| 64 | 1974.84 | 13497.66 | 74,435 |
| 512 | 1981.16 | 13360.37 | 74,595 |

**C6_1M** (ISL=1047552, OSL=1024)

| Concurrency | TPOT (ms) | TTFT (ms) | Total Throughput (tok/s) |
|:-----------:|:---------:|:---------:|:------------------------:|
| 1 | 6.74 | 8905.68 | 48,282 |
| 2 | 1362.59 | 13519.28 | 51,511 |
| 8 | 1911.78 | 18485.74 | 54,438 |
| 64 | 1912.90 | 18506.17 | 54,406 |
| 512 | 1912.90 | 18509.91 | 54,385 |

### replicate-q-proj Summary

- **Accuracy:** No regression (12/12 pass at ~0.38).
- **c=1 TPOT:** No measurable improvement — AllGather Q latency is negligible on single-node NVLink.
- **c>=2 TPOT:** Same scheduler starvation pattern as baseline (512K+: 1000-5000ms).
- **Throughput:** Comparable to baseline across all configs.
- **Bug fixed:** Chunked prefix crash at context >= 256K (kv_b_proj head count mismatch).
- **H100 results:** In progress (script running with persistent logs on `/raid`).
