# DCP A2A Optimization — Full Findings

**Date:** 2026-03-15
**Branch:** `htphan/helix_a2a_rebased_main_fe294904c9`
**Model:** DeepSeek-V2-Lite (15.7B, MLA, DCP8 on H100 + B200)

---

## 1. Code Changes Implemented (Merged)

### 1a. Remove redundant `.contiguous()` in CUDA graph path
**File:** `python/sglang/srt/layers/attention/dcp_a2a.py:236-239`

The Triton `_dcp_lse_combine_kernel` uses explicit strides, so non-contiguous views from `recv_buf[:, :B, :, :]` work correctly. Removes 2 D2D copy kernels per layer × 27 layers = 54 fewer GPU kernels per decode step.

### 1b. Per-bs CUDA graph buffer allocation
**File:** `python/sglang/srt/layers/attention/flashattention_backend.py`

- `init_cuda_graph_state()`: Changed from single max_bs=512 buffer to `dcp_cuda_graph_buffers_per_bs = {}` dict
- `_alloc_dcp_a2a_buffers_for_bs(bs)`: New method — lazily allocates right-sized `[N, bs, ...]` buffers during CUDA graph capture
- `init_forward_metadata_capture_cuda_graph()`: Calls `_alloc_dcp_a2a_buffers_for_bs(bs)` for each captured batch size
- `init_forward_metadata_replay_cuda_graph()`: Points `self.dcp_cuda_graph_buffers` to correct per-bs buffer

At B=1, NCCL `all_to_all_single` now transfers `N×1×H_per_rank×D` instead of `N×512×H_per_rank×D`.

### 1c. Fused output+LSE into single all_to_all
**File:** `python/sglang/srt/layers/attention/dcp_a2a.py:187-280`

Pack output (bf16) + LSE (fp32 reinterpreted as 2×bf16) into a single `[N, bs, H_per_rank, D+2]` buffer. One `all_to_all_single` call instead of two. Halves NCCL SendRecv calls per decode step (1512→756 confirmed by profiling).

**Buffer layout change** in `_alloc_dcp_a2a_buffers_for_bs()`:
- Old: `send_output [N,bs,H,D]`, `recv_output [N,bs,H,D]`, `send_lse [N,bs,H]`, `recv_lse [N,bs,H]`
- New: `send_combined [N,bs,H,D+2]`, `recv_combined [N,bs,H,D+2]`, `send_lse [N,bs,H]` (staging), `recv_lse [N,bs,H]` (staging)

### 1d. DCP scheduling pressure cap
**File:** `python/sglang/srt/managers/schedule_policy.py:474-477`

In `PrefillAdder.rem_total_tokens`, divide `available_and_evictable` by `dcp_size` when DCP > 1. Creates same memory pressure as TP to help prevent decode starvation.

**Note:** This alone doesn't fix the TPOT degradation because the scheduler's `prefill-first` priority still dominates. The cap only matters when the pool is near full.

---

## 2. Profiling Results (H100 + B200)

### 2a. Per-layer NCCL kernel breakdown (H100, 128K, c=1, ~27 layers × 8 profiled steps)

| NCCL Primitive | Kernel Type | Count | Total (ms) | Avg (μs) |
|---|---|---:|---:|---:|
| A2A SendRecv (fused, 1 call/layer) | `ncclDevKernel_SendRecv` | 756 | 4.72 | 6.2 |
| A2A SendRecv (old, 2 calls/layer) | `ncclDevKernel_SendRecv` | 1512 | 8.58 | 5.7 |
| AG_RS AllGather (LSE) | `ncclSymkDevKernel_AllGather_LLMC` | 432 | 1.36 | 3.1 |
| AG_RS ReduceScatter | `ncclSymkDevKernel_ReduceScatter_LL` | 216 | 0.69 | 3.2 |
| Shared Q AllGather | `ncclSymkDevKernel_AllGather_LLMC` | 216 | 0.66 | 3.1 |
| Triton dcp_lse_combine | kernel | 216 | 0.33 | 1.5 |
| Triton correct_attn | kernel | 216 | 0.31 | 1.4 |

### 2b. Root cause: A2A uses generic NCCL, AG_RS uses symm-mem

| | A2A DCP comm overhead | AG_RS DCP comm overhead |
|---|---:|---:|
| **H100** | 4.72ms (SendRecv) + 0.33ms (Triton) = **5.05ms** | 1.36ms (AG) + 0.69ms (RS) + 0.31ms (Triton) = **2.36ms** |
| **B200** | 2.64ms + 0.42ms = **3.06ms** | 2.06ms + 0.98ms + 0.35ms = **3.39ms** |

- `ncclSend/ncclRecv` always produces `ncclDevKernel_SendRecv` (~6μs) — **no symm-mem path exists**
- `ncclAllGather`/`ncclReduceScatter` use `ncclSymkDevKernel_*` (~3μs) when buffers are from symm-mem pool
- Allocating A2A buffers from symm-mem pool (tested) does NOT change the kernel — symm-mem only applies to collective operations, not point-to-point Send/Recv

### 2c. Triton kernel is NOT the bottleneck

`dcp_lse_combine_kernel` avg 1.5μs per call, total 0.33ms per decode step (<3% of DCP overhead). Even 10× faster would save only 0.3ms. The bottleneck is NCCL primitive latency.

---

## 3. Benchmark Results Summary

### 3a. Accuracy (GSM8K, 1319 questions)

| Variant | H100 | B200 |
|---|:---:|:---:|
| Baseline (DCP4, pre-opt) | 0.381-0.385 | 0.386-0.393 |
| DCP8 A2A per-bs opt | 0.376 | 0.387 |
| DCP8 A2A fused | 0.381 | 0.393 |
| DCP8 A2A fused + mixed-chunk | **0.002** (BROKEN) | not tested |

### 3b. H100 Performance: A2A fused vs AG_RS vs TP8

**128K (ISL=130048)**

| c | TP8 | AG_RS | A2A fused | Winner |
|:-:|:---:|:-----:|:---------:|:------:|
| 1 | 13.82ms | 4.19ms | 4.30ms | DCP (3× better than TP) |
| 4 | 11.46ms | 22.87ms | **17.20ms** | TP TPOT, **A2A throughput** (81K vs 69K) |
| 8 | 8.06ms | 38.05ms | **29.66ms** | TP TPOT, **A2A throughput** (81K vs 69K) |
| 64 | 9.13ms | 36.31ms | 32.96ms | TP TPOT, **A2A throughput** (82K vs 67K) |

**256K — A2A dominates AG_RS**

| c | AG_RS | A2A fused | A2A improvement |
|:-:|:-----:|:---------:|:---:|
| 1 | 133.96ms | **5.07ms** | 96% (AG_RS anomalous) |
| 8 | 1,062ms | **364ms** | 66%, throughput 84K vs 28K |
| 64 | 1,252ms | **364ms** | 71%, throughput 85K vs 6K (15×) |

**512K + 1M — Scheduler-dominated, AG_RS degrades more gracefully**

| Context | c=1 | c=8 AG_RS | c=8 A2A | Notes |
|---|:---:|:---:|:---:|---|
| 512K | ~6ms (both) | 889ms | 4,595ms | A2A 5× worse |
| 1M | ~6ms (both) | 3,064ms | 8,939ms | A2A 3× worse |

### 3c. B200 Performance (similar pattern)

- 128K: A2A matches AG_RS at all concurrencies (~96K tok/s)
- 256K: A2A slightly worse than AG_RS at c>=8 (171ms vs 97ms)
- 512K/1M: Both scheduler-dominated

---

## 4. Mixed-Chunk + DCP Investigation

### 4a. Why TPOT degrades at c>=2

The scheduler has a **prefill-first** policy (`scheduler.py:2099-2108`). With DCP's 8× effective KV pool, memory pressure never triggers → prefill runs continuously → decode starved.

`--enable-mixed-chunk` merges decode tokens into prefill batches so decode runs alongside prefill. But DCP's extend path is incompatible with mixed batches.

### 4b. What hangs

**File:** `model_runner.py:2399-2532` — DCP extend preparation

The DCP extend path:
1. Allocates `dcp_kv_buffer = torch.empty(seq_lens_sum, kv_dim)` — proportional to total sequence lengths
2. Builds `dcp_kv_indices` — maps tokens with DCP interleaving
3. Does `all_gather_into_tensor` of prefix KV from all DCP ranks

For a **mixed batch** with 1 prefill chunk (32K) + N decode requests (128K seq_len each):
- `seq_lens_sum` ≈ 32K + N×128K = huge
- `dcp_kv_buffer` allocation either OOMs or takes too long → watchdog timeout (300s)

### 4c. Fix attempt and results

**Approach:** Skip DCP extend prep for MIXED mode, let attention use standard `page_table`.

**Changes tried:**
- `model_runner.py:2403`: `if get_dcp_world_size() > 1 and not forward_batch.forward_mode.is_mixed():`
- `deepseek_v2.py:1870`: Guard `dcp_kv_buffer` access with `getattr(forward_batch, 'dcp_kv_buffer', None) is not None`
- `deepseek_v2.py:1601`: Guard `dcp_local_prefix_kv_indices` access similarly

**Results:**
- Server no longer hangs ✓
- TPOT at c=4: **5.79ms** (was 17.20ms, -66%) ✓
- TPOT at c=16: **6.11ms** (was 32.90ms, -81%) ✓
- **Accuracy: 0.002** (was 0.38) ✗ — CATASTROPHIC

**Why accuracy breaks:** The DCP extend path does an essential all_gather of interleaved KV cache from all ranks. Without it, each rank sees only its local 1/N of the KV cache during prefill attention → wrong attention output → garbage tokens.

### 4d. Correct fix required

The proper mixed-chunk + DCP fix must:

1. **Split the mixed batch** into extend requests and decode requests
2. **For extend requests:** Run the full DCP extend path (all_gather KV, build dcp_kv_indices, etc.) with only the extend requests' `seq_lens_sum`
3. **For decode requests:** Run the DCP decode path (Q all-gather → local attention with dcp_page_table → A2A reduce)
4. **Merge results** from both sub-batches

**Files that need changes:**
- `python/sglang/srt/model_executor/model_runner.py:forward_extend()` — Split DCP prep for extend vs decode requests
- `python/sglang/srt/models/deepseek_v2.py:forward_absorb_prepare()` — Handle mixed batch in MLA DCP path
- `python/sglang/srt/models/deepseek_v2.py:forward_absorb_core()` — Route extend vs decode through correct DCP attention path
- `python/sglang/srt/layers/attention/flashattention_backend.py:init_forward_metadata()` — Set up both extend and decode DCP metadata for mixed batch

**Key constraint:** All DCP ranks must execute the same NCCL operations (same number of all_gather/all_to_all calls) to avoid deadlocks.

---

## 5. Optimization Paths Not Taken

### 5a. Replace SendRecv with AllGather + local permute
Use `ncclAllGather` (symm-mem, 3μs) instead of `ncclSend/ncclRecv` (6μs) for the A2A data exchange. Trades bandwidth (N× more data) for latency. At B=1 the data is tiny so bandwidth doesn't matter. **Rejected by user** — changes communication pattern too much.

### 5b. Triton kernel optimization
The `dcp_lse_combine_kernel` is already 1.5μs — <3% of DCP overhead. Possible improvements (online softmax, vectorized LSE load) would save <0.2μs. Not worth the effort.

### 5c. Async Q all-gather overlap
Overlap the Q all-gather with other computation using async NCCL. Requires async NCCL infrastructure changes. Not attempted.

---

## 6. Current State of Code

**Changes committed (working, accuracy-verified):**
- `dcp_a2a.py`: Fused all_to_all, removed `.contiguous()`, `_lse_pack_dim()` helper
- `flashattention_backend.py`: Per-bs buffer allocation via `_alloc_dcp_a2a_buffers_for_bs()`
- `schedule_policy.py`: DCP scheduling cap (`available_and_evictable //= dcp_size`)
- `scheduler.py`: Pass `dcp_size` to `PrefillAdder`

**Changes reverted (broken accuracy):**
- `model_runner.py`: Skip DCP extend for MIXED mode — reverted
- `deepseek_v2.py`: Guard DCP KV gather for MIXED mode — reverted

---

## 7. Key Numbers to Remember

| Metric | Value |
|---|---|
| NCCL SendRecv avg latency | 5.8μs (H100), 6.1μs (B200) |
| NCCL SymkAllGather avg latency | 3.1μs (H100), 4.5μs (B200) |
| DCP comm overhead per decode step (A2A fused) | 5.05ms (H100), 3.06ms (B200) |
| DCP comm overhead per decode step (AG_RS) | 2.36ms (H100), 3.39ms (B200) |
| Triton combine kernel | 1.5μs per call, 0.33ms per step |
| A2A fused best TPOT vs AG_RS (128K c=4) | 17.20ms vs 22.87ms (-25%) |
| A2A fused throughput advantage (128K c>=4) | 81K vs 66K tok/s (+22%) |
| A2A fused best context (256K c>=8) | 364ms vs 1063ms (-66%), 85K vs 28K tok/s (3×) |
| Mixed-chunk DCP fix TPOT (128K c=4) | 5.79ms (but accuracy broken) |
