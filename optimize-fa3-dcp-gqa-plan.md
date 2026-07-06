# Plan: Optimize FA3 DCP for GQA to Match FlashInfer DCP Performance

## Context

**Problem**: FA3 (FlashAttention3) DCP decode for GQA models is significantly slower than FlashInfer DCP on the same hardware (H100 NVL 8-GPU). Benchmarking Qwen3-235B with 32K/4K workload shows:
- FlashInfer DCP2: 64.95 tok/s at cc=1 (TPOT 14.5ms)
- FA3 DCP2: 49.52 tok/s at cc=1 (TPOT 19.4ms) — **24% slower**
- FA3 DCP also crashes at cc=32 (illegal memory access) while FlashInfer survives to cc=96+

MLA DCP on FA3 performs comparably to FlashInfer, proving the issue is specific to how GQA DCP is structured in the FA3 backend, not a fundamental FA3 limitation.

**Primary test model**: Qwen3-235B-A22B-Instruct-2507 (MoE, 4 KV heads, 94 layers)

## Code Branches

| Branch | Base | Key Commits | Description |
|--------|------|-------------|-------------|
| `htphan/repro-fenp-dcp-gqa` | sglang main | `5008279` (FlashInfer DCP), `74a94ec` (FA3 DCP) | **Target branch for this plan.** Has both FlashInfer DCP (PR #14982 squashed) and FA3 DCP side by side for comparison. |
| `htphan/improve-tpa` | sglang main | `b8473d8` (FA3 DCP), `74381ad` (TPA), `bb0a64f`/`4006989` (Triton A2A kernels), `cb67b95` (S13 bench) | Old implementation branch. Has FA3 DCP + TPA + Triton A2A pack/unpack/combine fusion + profiling scripts + S1-S13 benchmark scenarios. |

### Key commits on `htphan/improve-tpa` (old implementation):
- `b8473d8ac` — FA3 DCP with A2A and AG+RS backends (same as `74a94ec` on repro branch)
- `74381ada9` — Phase-1 TPA layout, decode merge, and validation (adds `--attention-tensor-parallel-size`)
- `26ccbfc5e` — TPA correctness fixes, DCP CUDA graph fixes, Helix ReduceScatter, Qwen3 MoE support
- `bb0a64ff8` — Triton A2A pack kernel (fuses permute + 3 copy_ ops into 1 kernel, +2-4%)
- `40069897c` — Fused unpack + LSE combine Triton kernel (eliminates intermediate recv_lse buffer)
- `438f843ef` — Skip unnecessary `lse_base2` division for A2A paths
- `f3def8538` — S11-S12 decode-focused benchmark scenarios (prefix caching, Zhao reproduction)
- `cb67b95ea` — S13 PR #14982 reproduction scenario

### Key commits on `htphan/repro-fenp-dcp-gqa` (target branch):
- `500827972` — Squashed FENP FlashInfer DCP for GQA (PR #14982)
- `74a94ec48` — FA3 DCP implementation (same code as `b8473d8` on improve-tpa)
- `6496fa123` — Pool leak checker fix for DCP
- `9c869dd0f` — Simplified bench script for PR comparison

### Triton optimizations from `htphan/improve-tpa` to port forward:
The Triton kernels from `bb0a64f` and `4006989` (in `dcp_a2a.py`) should be ported to the repro branch as they reduce A2A overhead by 2-4%. Key kernels:
- `_pack_a2a_send_kernel` — fuses permute + LSE packing into single kernel
- `_fused_unpack_combine_kernel` — fuses unpack + LSE-weighted combine into single kernel

## Root Causes (3 gaps, ordered by impact)

### Gap 1: Per-layer Q All-Gather + DCP Reduce Inside Backend (Highest Impact)
- **FA3 GQA** (`flashattention_backend.py:1203-1285`): Every layer does Q all-gather → attention → LSE normalize → DCP reduce (A2A or AG+RS) — all inside `forward_decode`
- **FlashInfer** (`flashinfer_backend.py:944-979`): Same structure but uses symmetric memory for Q clone and attention output, enabling zero-copy NCCL
- **MLA DCP** (`forward_mla.py:432-528`): Model code all-gathers Q once, creates a wider `RadixAttention` (`num_heads * dcp_size`), and backend just returns `(output, lse)` — model handles reduce. No Q all-gather in the backend.

### Gap 2: KV Index Construction (~9 PyTorch ops vs 1 Triton kernel)
- **FA3** (`flashattention_backend.py:2434-2492`, `_init_dcp_decode_metadata`): `arange` + integer arithmetic + `clamp` + fancy indexing + `copy_` = ~9 CUDA kernel launches per step
- **FlashInfer** (`utils.py:57-93`): Single `create_flashinfer_kv_indices_for_dcp_triton` kernel with strided loads + divide

### Gap 3: A2A Buffer Pack/Unpack Overhead
- FA3 A2A: 5-6 extra copies per layer (reshape → pack output+LSE → A2A → unpack → Triton combine)
- FlashInfer AG+RS: ~2 copies per layer (symmetric memory wrapping eliminates extra copies)
- **Note**: Triton kernels from `htphan/improve-tpa` (`bb0a64f`, `4006989`) partially address this — port them first

## Implementation Plan

### Phase 1: Triton Kernel for DCP Page Table (replaces Gap 2)

**File**: `python/sglang/srt/layers/attention/utils.py`

Add `create_fa3_dcp_page_table_triton` kernel:
- Grid: `(B,)` — one program per batch element
- Per program: compute `local_seqlen`, then for each local page index, compute global page index via `dcp_rank + p * page_size * dcp_size`, load from `page_table[batch, global_page_idx]`, divide by `dcp_size`, store to `dcp_page_table`
- Also writes `dcp_cache_seqlens` in the same kernel

**File**: `python/sglang/srt/layers/attention/flashattention_backend.py`

Replace body of `_init_dcp_decode_metadata` (lines 2434-2492) with single Triton kernel call. Pre-allocated CUDA graph buffers (`decode_cuda_graph_metadata["dcp_page_table"]` and `["dcp_cache_seqlens"]`) are passed directly as output pointers.

### Phase 2: MLA-pattern Restructure for GQA DCP (addresses Gap 1)

Follow the proven MLA DCP pattern where Q is all-gathered at model level and the backend just returns `(output, lse)`.

**Step 2a — Backend: Return (output, lse) for GQA DCP**

**File**: `python/sglang/srt/layers/attention/flashattention_backend.py`

Modify GQA DCP path in `forward_decode` (lines 1115-1285):
- When `use_dcp` and Q is already full-width (detect via `q.shape[1] == layer.tp_q_head_num * dcp_size` after reshape), **skip Q all-gather** and **return `(output, lse_2d)` tuple** (same as MLA path at line 1386)
- When Q is local-width (backward compat), keep existing behavior (all-gather + reduce inside backend)

Detection logic:
```python
q_reshaped = q.view(-1, layer.tp_q_head_num, layer.head_dim)
q_already_gathered = (q_reshaped.shape[1] == layer.tp_q_head_num * self.dcp_size)
if use_dcp and not q_already_gathered:
    q_reshaped = get_dcp_group().all_gather(q_reshaped.contiguous(), dim=1)
```

And at the DCP reduce section:
```python
if use_dcp:
    # ... LSE normalization ...
    if q_already_gathered:
        # Model handles reduce — return (output, lse) like MLA
        return o.view(-1, H_out * layer.v_head_dim), lse_2d
    else:
        # Backend handles reduce (backward compat)
        if self.dcp_comm_backend == "a2a":
            o = dcp_a2a_lse_reduce(...)
        else:
            o = cp_lse_ag_out_rs(...)
```

**Step 2b — Model: All-gather Q and reduce at model level**

**File**: `python/sglang/srt/models/qwen3_moe.py` (and similar GQA models)

In `Qwen3MoeAttention.__init__`:
```python
if get_dcp_world_size() > 1:
    self.attn_for_dcp_decode = RadixAttention(
        self.num_heads * get_dcp_world_size(),  # wider Q
        self.head_dim, self.scaling,
        num_kv_heads=self.num_kv_heads,
        layer_id=layer_id, v_head_dim=self.head_dim,
    )
```

In `Qwen3MoeAttention.forward` (decode + DCP path):
```python
if forward_batch.forward_mode.is_decode() and get_dcp_world_size() > 1:
    q_3d = attn_output.view(-1, self.num_heads, self.head_dim)
    with use_symmetric_memory(get_dcp_group()):
        q_3d = q_3d.clone(memory_format=torch.contiguous_format)
    q_gathered = get_dcp_group().all_gather(q_3d, dim=1)
    q_flat = q_gathered.flatten(1)
    attn_output, lse = self.attn_for_dcp_decode(q_flat, k, v, forward_batch)
    # Reduce
    attn_output = attn_output.view(-1, self.num_heads * dcp_size, self.head_dim)
    if comm_backend == "a2a":
        attn_output = dcp_a2a_lse_reduce(attn_output, lse, get_dcp_group(), is_lse_base_on_e=True)
    else:
        attn_output = cp_lse_ag_out_rs(attn_output, lse, get_dcp_group(), is_lse_base_on_e=True)
else:
    attn_output = self.attn(attn_output, k, v, forward_batch)
```

### Phase 3: Symmetric Memory + Buffer Fixes (addresses Gap 3)

**File**: `python/sglang/srt/layers/attention/flashattention_backend.py`

For the backward-compat path (Q all-gather inside backend):
1. Wrap Q clone in symmetric memory (like FlashInfer line 945-946)
2. For AG+RS path, wrap output+LSE in symmetric memory before `cp_lse_ag_out_rs`
3. For CUDA graph mode, use `all_gather_into_tensor` with pre-allocated `q_gathered` buffer

**File**: `python/sglang/srt/layers/attention/dcp_a2a.py`

1. Port Triton kernels from `htphan/improve-tpa` (`bb0a64f`, `4006989`): `_pack_a2a_send_kernel`, `_fused_unpack_combine_kernel`
2. Accept `[H, B]` LSE layout directly (add `lse_transposed=True` parameter) to eliminate the `softmax_lse.T.contiguous()` copy in the backend

## Critical Files

| File | Changes |
|------|---------|
| `python/sglang/srt/layers/attention/utils.py` | New `create_fa3_dcp_page_table_triton` kernel |
| `python/sglang/srt/layers/attention/flashattention_backend.py` | Replace `_init_dcp_decode_metadata`, modify GQA DCP path to return (output, lse) when Q is pre-gathered |
| `python/sglang/srt/models/qwen3_moe.py` | Add `attn_for_dcp_decode`, all-gather Q + reduce at model level |
| `python/sglang/srt/layers/attention/dcp_a2a.py` | Port Triton kernels from improve-tpa, accept [H,B] LSE |

## Verification

### Accuracy
```bash
# Inside container on colossus
cd /sgl-workspace/sglang
# FA3 DCP2
python3 -m sglang.launch_server --model-path Qwen/Qwen3-235B-A22B-Instruct-2507 --tp 8 --dcp 2 --dcp-comm-backend a2a --attention-backend fa3 --enable-symm-mem --port 8188
python3 -m sglang.test.few_shot_gsm8k --parallel 128 --max-new-tokens 512
# Should match FlashInfer DCP2 accuracy (0.965-0.975)
```

### Performance
```bash
# Run S13 benchmark (PR #14982 reproduction)
bash benchmark/dcp/bench_tpa_gqa_serving.sh scenario13 perf
# FA3 DCP2 should match FlashInfer DCP2 within 5%:
#   cc=1 TPOT: target ~15ms (currently 19.4ms)
#   cc=96 throughput: should not crash (currently crashes at cc=32)
```

### Crash Investigation
The cc=32 crash (`moe_sum_reduce illegal memory access`) may be related to CUDA graph buffer corruption under DCP. Phase 1 (Triton kernel) and Phase 3 (proper symmetric memory + pre-allocated buffers) may fix it. If not, test with `--disable-cuda-graph` to isolate.
