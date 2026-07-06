# TPA Phase-1 Summary and Clean `ctx=1M` Decode Results

Date: 2026-03-30
Current local TPA commit: `5dff48333` (`Add phase-1 TPA layout, decode merge, and validation`)

## 1. What Changed For TPA

The current phase-1 TPA commit adds the minimum runtime surface needed to run TPA with DCP for standard MHA models, with Qwen2 as the validated model family.

Main changes:

- Added explicit TPA configuration and validation in server/runtime setup.
  - `--attention-tensor-parallel-size`
  - explicit `--dcp-size`
  - validation that phase-1 TPA requires `tp_size = attention_tp_size * dcp_size`
- Updated distributed initialization so attention TP groups and DCP groups are built with the TPA layout instead of assuming the old plain-TP layout.
- Added `dcp_layout.py` to build DCP-local decode metadata from the global page table.
- Updated the FlashAttention backend to support TPA-aware decode merge and output partitioning.
- Updated Qwen2 attention and decoder-layer wiring so attention can hand off a full-TP-shaped output shard to `o_proj` when TPA is enabled.
- Updated `LayerCommunicator` so the residual and layer handoff path matches the stronger TPA output layout.
- Added a safety guard to keep TPA off the fused RoPE+KV-store fast path until that kernel path is validated for the narrower TPA layout.
- Added regression coverage for:
  - server-arg validation
  - DCP local page-table construction
  - Qwen2 TPA model wiring
  - TPA communicator behavior

In short: the commit is not just a flag addition. It changes the runtime layout contract from argument parsing, to distributed groups, to decode attention merge, to model-layer handoff.

## 2. Decode Optimization We Added For TPA

The main decode optimization in the current TPA commit came from Hopper `nsys` profiling, not from guesswork.

### 2.1 What `nsys` showed

We profiled the Hopper decode path and found that the first large TPA slowdown was not mainly inside the attention kernel. The clearest gap was after attention, in the communicator path that was converting a TPA-local layout back into the normal full-TP layout.

The key `nsys` signal was:

- TPA `layer_comm.postprocess_layer`: about `29.6 ms`
- DCP `layer_comm.postprocess_layer`: about `0.7 ms`

So the first real decode bottleneck was a post-attention layout-conversion tax.

### 2.2 What optimization we implemented from that result

Based on that profiling result, we changed the decode path so attention hands off a full-TP-friendly output earlier, instead of letting the communicator repair that layout later.

The key ideas are:

- Keep attention sharded by attention TP instead of forcing the plain-DCP decode path to all-gather query heads before attention.
- Let attention return only the output head slice that the local `o_proj` shard actually needs.
- Use a TPA-aware DCP merge path after decode attention.
- Hand the merged attention result to `o_proj` in a full-TP-friendly layout so the downstream communicator path does not pay the old post-attention conversion tax.

Concretely, the optimization includes:

- Output-head partition contract:
  - `Qwen2Attention` tells `RadixAttention` which output head slice belongs to this rank.
  - The attention backend returns that contracted output shape instead of always returning the full local attention width.
- TPA-aware decode merge in the FlashAttention backend:
  - when ranks own disjoint output-head slices, use `dcp_a2a_lse_reduce(...)`
  - when ranks already hold replicated heads, use `cp_lse_ag_out_allreduce(...)`
- Dedicated CUDA-graph buffers for the TPA DCP A2A decode path.
- Updated layer communication so the post-attention residual path stays aligned with the TPA handoff contract.

### 2.3 What changed after the optimization

The post-attention tax essentially disappeared in the trace:

- TPA `layer_comm.postprocess_layer`: `29.554 ms -> 0.709 ms`
- DCP `layer_comm.postprocess_layer`: `0.676 ms -> 0.704 ms`

So the decode optimization we added for TPA was not a generic micro-optimization. It was a targeted fix for the exact post-attention bottleneck that `nsys` exposed.

## 3. What Is Left To Implement For TPA

### 3.1 A working FA4-backed TPA path

Current phase-1 TPA is explicitly gated to `fa3` / `fa4`, but in practice we do not yet have a validated FA4-backed TPA path.

What is known:

- `fa3` is the working backend for the current Hopper TPA path.
- On Blackwell, the FA4 path failed because the backend did not return a usable LSE tensor for the DCP-style decode merge path.
- So Blackwell is still blocked for the current TPA implementation.

That means one of the next backend tasks is:

- make FA4 work for the TPA/DCP decode merge contract, or
- add a FlashInfer-backed TPA decode path for normal Qwen-style MHA models

### 3.2 Backend support beyond the current FlashAttention path

FlashInfer-backed TPA is not implemented yet for normal Qwen-style MHA.

So today:

- TPA is effectively a FlashAttention-path feature
- there is no working generic FlashInfer TPA path yet

### 3.3 Broader model-family support

The current implementation assumes model families that already shard QKV projections with `get_attention_tp_size()`.

Qwen2 is the validated path. Other models need explicit verification that:

- their QKV layout matches the phase-1 TPA assumptions
- their `o_proj` sharding can consume the TPA handoff contract correctly
- their communicator path does not need model-specific adjustments

### 3.4 More decode tuning

There is still room to profile and tune decode further. The most plausible remaining decode-side hotspots are:

- plain DCP query all-gather before decode attention
- TPA A2A + LSE reduce path
- communicator work around attention / MLP handoff
- DCP local decode metadata construction

## 4. Decode Comparison Results: Clean Whole-Checkout Runs At `ctx=1M`

These are the clean whole-checkout runs from the current branch checkout, and they replace the earlier `c=32` numbers that were taken before the benchmark snapshot was resynced.

Shared run characteristics:

- model: `Qwen/CodeQwen1.5-7B-Chat`
- model attention layout: 32 attention heads, 4 KV heads
- whole TPA branch checkout mounted into the container
- warmed identical prompt shape
- decode-focused serving measurement

### 4.1 `c=32`

For `c=32`, all three modes landed:

- `output_seq_len_avg = 769`
- `input_seq_len_avg = 1047452`
- `osl_mismatch_count_avg = 96`

So this point is apples-to-apples across modes.

| Mode | Request Throughput | Output Tok/s | ITL Avg (ms) | ITL p50 (ms) | Output Len Avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pure_tp` | 0.02849 | 21.91 | 1392.03 | 1422.77 | 769 |
| `dcp2_a2a` | 0.05637 | 43.35 | 673.09 | 706.77 | 769 |
| `tpa2_dcp4_a2a` | 0.05682 | 43.69 | 665.44 | 700.62 | 769 |

Relative read:

- `dcp2_a2a` and `tpa2_dcp4_a2a` are both about 2x better than `pure_tp` on this point.
- TPA is slightly ahead of DCP in this clean `c=32` rerun.
  - about 1.1% on ITL average
  - about 0.9% on ITL p50

### 4.2 `c=64`

For `c=64`, all three modes landed:

- `output_seq_len_avg = 769`
- `input_seq_len_avg = 1047452`
- `osl_mismatch_count_avg = 192`

So this point is also apples-to-apples across modes.

| Mode | Request Throughput | Output Tok/s | ITL Avg (ms) | ITL p50 (ms) | Output Len Avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pure_tp` | 0.03266 | 25.11 | 2421.45 | 2484.46 | 769 |
| `dcp2_a2a` | 0.06033 | 46.40 | 1246.47 | 1312.12 | 769 |
| `tpa2_dcp4_a2a` | 0.05908 | 45.43 | 1278.41 | 1345.28 | 769 |

Relative read:

- `dcp2_a2a` and `tpa2_dcp4_a2a` are both still about 2x better than `pure_tp`.
- DCP is slightly ahead of TPA at `c=64`.
  - about 2.5% on ITL average
  - about 2.5% on ITL p50

## 4.3 Why TPA Stays Near Parity With DCP Here

There are two main reasons.

### A. TPA trades shorter KV length for more heads per rank

At fixed `tp=8` for CodeQwen 7B, TPA does not simply "do less decode work."

The rough effect is:

- plain `dcp2_a2a` decode sees more KV positions per rank
- `tpa2_dcp4_a2a` sees fewer KV positions per rank
- but TPA also gives each rank more attention heads to process

So the first-order decode attention work stays in a similar compute class instead of dropping dramatically.

This is the main structural reason TPA stays near parity instead of clearly pulling away.

### B. The remaining differences are secondary, and can swing small wins either way

TPA does avoid one plain-DCP overhead: the pre-attention query all-gather.

But the remaining differences between `dcp2_a2a` and `tpa2_dcp4_a2a` do not look large enough, by themselves, to dominate the result:

- the A2A + LSE reduce path after attention is shared DCP merge work, not a TPA-only cost
- the row-parallel combine after `o_proj` is also mostly shared layer-stack work, not a special TPA penalty
- output-partition coordination around attention and `o_proj` is mostly metadata and contract management, and should be a small effect in the intended fast path

So the best read is:

- the head-for-sequence trade in section A is the primary reason TPA does not show a large structural win over DCP
- the remaining merge and layout differences are secondary effects that can swing a small edge either way depending on the concurrency point

That is exactly what the clean runs show:

- at `c=32`, TPA is slightly ahead
- at `c=64`, DCP is slightly ahead
- in both cases, TPA and DCP are in the same decode performance class

## Bottom Line

Current status is:

- Phase-1 TPA decode runtime exists and is working.
- The main decode optimization for TPA is already in the commit.
- On clean whole-checkout `ctx=1M` runs, TPA and DCP are both much better than pure TP.
- TPA and DCP are near parity overall: TPA is slightly ahead at `c=32`, while DCP is slightly ahead at `c=64`.
- The biggest remaining missing pieces are a working FA4-backed path, broader backend/model coverage, and more decode tuning.