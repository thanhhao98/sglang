# Investigation: DCP8 ag_rs TPOT Degradation at 512K+ Context

## Context

B200 DCP8 ag_rs shows catastrophic TPOT degradation at 512K context:
- c=1: **6.76ms** (normal)
- c=2: **793ms** (117x worse)
- c>=8: **2400-4200ms** (350-600x worse)

TP8 at same context stays <41ms. vLLM DCP8 ag_rs stays ~72ms at 512K c=64.
This is SGLang-specific. Root cause likely in the ag_rs communication path (`cp_lse_ag_out_rs` in `utils.py:641-662`).

**Suspect code path:** Each decode step calls:
1. `all_gather()` on LSE — blocking NCCL collective
2. Triton kernel `correct_attn_out()` — scales with batch size
3. `reduce_scatter_along_dim()` — includes `movedim().contiguous()` copy + `torch.empty()` allocation + blocking NCCL collective

---

## Plan

### Step 1: Install nsys in existing container (no new image needed)

nsys is NOT in the current `sglang-bench` container. Install it directly — faster than rebuilding the image.

```bash
ssh colossus_b200_1 'sudo docker exec sglang-bench bash -c "
apt update && \
apt install -y --no-install-recommends gnupg && \
echo \"deb http://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64 /\" | tee /etc/apt/sources.list.d/nvidia-devtools.list && \
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
apt update && \
apt install -y nsight-systems-cli && \
pip install nvtx
"'
```

Verify: `docker exec sglang-bench nsys --version`

Also install on H100 if we want to compare.

### Step 2: Profile DCP8 ag_rs at 512K — two conditions

**Condition A: c=1 (fast, TPOT=6.76ms)**
**Condition B: c=8 (slow, TPOT=2466ms)**

For each condition, launch server with NVTX markers under nsys, then run bench_serving.

#### Server launch (inside container):
```bash
# DCP8 ag_rs, 512K context, with NVTX + nsys
nsys profile --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -o /output/profile_agrs_512K_c1 \
  python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V2-Lite \
    --tp 8 --context-length 524288 \
    --mem-fraction-static 0.75 \
    --chunked-prefill-size 131072 \
    --dcp-size 8 --dcp-comm-backend ag_rs \
    --enable-layerwise-nvtx-marker \
    --disable-cuda-graph
```

Note: `--disable-cuda-graph` is required so NVTX markers are emitted for all kernels.

#### Trigger profiling + run benchmark (from another terminal in container):
```bash
# Wait for server ready, then start profiling for 5 steps
curl -X POST http://127.0.0.1:30000/start_profile \
  -H "Content-Type: application/json" \
  -d '{"start_step": 2, "num_steps": 5, "activities": ["CUDA_PROFILER"]}'

# Condition A: c=1
python3 -m sglang.bench_serving --backend sglang \
  --model deepseek-ai/DeepSeek-V2-Lite \
  --num-prompts 3 --random-input-len 523264 --random-output-len 1024 \
  --random-range-ratio 0.0 --max-concurrency 1 --disable-tqdm
```

Then kill server, repeat with `--max-concurrency 8` for Condition B (separate nsys output file).

#### Key files to modify/create:
- `/output/run_profile_agrs.sh` — script automating both conditions on B200

### Step 3: Extract and compare nsys reports

Use SGLang's built-in analysis tool:
```bash
python3 /sgl-workspace/sglang/examples/profiler/nsys_profile_tools/gputrc2graph.py \
  --in_file /output/profile_agrs_512K_c1.nsys-rep,sglang,ds,<runtime_c1> \
            /output/profile_agrs_512K_c8.nsys-rep,sglang,ds,<runtime_c8> \
  --out_dir /output/profile_comparison \
  --title "DCP8 ag_rs 512K: c=1 vs c=8"
```

This produces HTML visualization + CSV with kernel-level time breakdown.

Additionally, use nsys stats for raw NCCL timing:
```bash
nsys stats --report cuda_gpu_trace /output/profile_agrs_512K_c1.nsys-rep > /output/profile_c1_kernels.txt
nsys stats --report cuda_gpu_trace /output/profile_agrs_512K_c8.nsys-rep > /output/profile_c8_kernels.txt
```

### Step 4: What to look for in profiles

Compare c=1 vs c=8 for:

1. **NCCL collective time** — `all_gather` and `reduce_scatter` kernel durations
   - If these blow up: communication bottleneck (tensor size scales with batch)
2. **Memory allocation time** — `torch.empty()` calls in `reduce_scatter_along_dim()` (`parallel_state.py:772`)
   - If significant: hot-path allocation overhead
3. **`movedim().contiguous()` copy time** — data layout transformation (`parallel_state.py:765`)
   - If significant: unnecessary memcpy in critical path
4. **Triton kernel `_correct_attn_cp_out_kernel`** — grid is `(B, H, 1)`, scales with batch
   - If significant: compute bottleneck in LSE correction
5. **Gaps between kernels** — CPU overhead / synchronization stalls
   - If large gaps: Python/framework overhead between NCCL calls
6. **Symmetric memory context manager** — `use_symmetric_memory()` overhead per call

### Step 5: Profile TP8 at same conditions for baseline

Run same nsys profiling with TP8 config at 512K, c=1 and c=8, to compare communication patterns:
```bash
# Same server launch but without DCP flags
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V2-Lite \
  --tp 8 --context-length 524288 \
  --mem-fraction-static 0.75 \
  --enable-layerwise-nvtx-marker \
  --disable-cuda-graph
```

This gives a baseline: if TP8 c=8 is fine but DCP8 ag_rs c=8 is 100x slower, the bottleneck is definitively in the ag_rs path.

---

## Critical Files

| File | Role |
|------|------|
| `python/sglang/srt/layers/attention/utils.py:641-662` | `cp_lse_ag_out_rs()` — AG+RS orchestrator |
| `python/sglang/srt/layers/attention/utils.py:479-557` | Triton kernel `_correct_attn_cp_out_kernel` |
| `python/sglang/srt/distributed/parallel_state.py:747-781` | `reduce_scatter_along_dim()` — movedim + alloc + NCCL |
| `python/sglang/srt/distributed/parallel_state.py:890-962` | `all_gather()` implementation |
| `python/sglang/srt/distributed/device_communicators/pynccl.py:196-330` | NCCL all_gather/reduce_scatter |
| `python/sglang/srt/layers/attention/dcp_a2a.py:187-251` | A2A path (for comparison) |
| `examples/profiler/nsys_profile_tools/gputrc2graph.py` | nsys analysis tool |

---

## Hypothesized Root Causes (Pre-Profiling)

Based on code analysis and the symptom pattern (c=1 fine, c>=2 catastrophic), here are ranked hypotheses:

### Hypothesis 1: NCCL Collective Tensor Size Explosion (Most Likely)

**Code:** `cp_lse_ag_out_rs()` in `utils.py:641-662`

At 512K context with DCP8, each rank holds `context_length / dcp_size = 64K` tokens of KV. During decode, each new token requires attending to all 512K tokens across 8 ranks.

The ag_rs path does:
- `all_gather(cp_attn_lse)` — gathers LSE of shape `[B, H]` across 8 ranks → `[8, B, H]`
- `reduce_scatter(out)` — scatters corrected output of shape `[8*B, H, D]` → `[B, H, D]`

At c=1, B=1: these are small tensors. At c=8, B=8: tensor sizes 8x larger, but NCCL collectives should scale sub-linearly, NOT 350x. **So raw NCCL time alone probably doesn't explain 350x.**

**Why it might still matter:** The all_gather output `[8, B, H]` is used by the Triton kernel which runs grid `(B, H, 1)`. With B=8 and H=many_heads (DeepSeek-V2-Lite has 16 heads), the Triton kernel runs 8x more blocks. But still not 350x.

### Hypothesis 2: Scheduler Serialization / Batched Prefill Blocking Decode (Most Likely)

**The real smoking gun:** TPOT measures *mean* across all tokens. At c=8, 8 requests are in-flight. With 512K context, each prefill takes ~5.4s (see TTFT at c=1).

If the scheduler processes prefill chunks for new requests **between decode steps**, then:
- At c=1: 1 prefill, then continuous decode — TPOT is pure decode latency
- At c=8: multiple prefills interleave with decode — decode steps get **starved** while prefill chunks run

The TPOT=2466ms at c=8 ≈ the time to process one chunked prefill iteration (131072 tokens at ~20ms/token ≈ 2.6s). This matches almost exactly.

**Key question:** Does the scheduler interleave chunked prefill with decode differently for DCP vs TP? If DCP ag_rs has extra synchronization that forces prefill and decode to serialize across ranks, this could explain why TP8 (TPOT=37ms at c=8) is fine but DCP8 ag_rs (TPOT=2466ms at c=8) is not.

**Evidence supporting this:**
- c=1 TPOT is excellent (6.76ms) — no prefill interference
- c=2 TPOT jumps to 793ms — second request's prefill blocks first request's decode
- TTFT at c=8 is 11876ms ≈ 2x TTFT at c=1 (5427ms) — prefills are serialized

### Hypothesis 3: `movedim().contiguous()` Memory Copy in Hot Path

**Code:** `reduce_scatter_along_dim()` in `parallel_state.py:765`
```python
input_tensor = input_.movedim(0, dim).contiguous()  # forces memcpy
```

This is called **every decode step, every layer**. At 512K with DCP8:
- Input shape: `[8*B, H, D]` where B=batch_size
- `movedim(0, dim)` transposes + `contiguous()` allocates new tensor and copies

For a 27-layer model (DeepSeek-V2-Lite), this is 27 memcpy operations per decode step. With B=8 at 512K, each tensor is larger, and the copy could become significant relative to the actual NCCL time.

**A2A comparison:** `dcp_a2a.py` uses `all_to_all_single()` which operates on pre-shaped tensors without dimension manipulation. No movedim+contiguous needed.

### Hypothesis 4: `torch.empty()` Allocation on Every Decode Step

**Code:** `parallel_state.py:772-776`
```python
output_tensor = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
```

Called every decode step, every layer. PyTorch CUDA allocator is generally fast, but under memory pressure at 512K context, the allocator may need to defragment or wait for prior operations.

**Less likely** to cause 350x slowdown alone, but combined with hypothesis 3, the per-step overhead compounds across 27 layers.

### Hypothesis 5: Symmetric Memory / NCCL Stream Synchronization

**Code:** `use_symmetric_memory()` context manager wraps both the `contiguous()` call and the `torch.empty()` call.

If symmetric memory requires stream synchronization (e.g., waiting for prior NCCL ops to complete before allocating), this could serialize operations that should be pipelined.

### Hypothesis 6: Disable CUDA Graph Penalty for DCP ag_rs

The benchmark was run **with** CUDA graphs enabled (default). CUDA graphs capture the execution and replay it. But if the ag_rs path has dynamic shapes (batch size changes) or dynamic allocations (`torch.empty` with variable shapes), CUDA graph capture may fail silently and fall back to eager mode, adding Python overhead.

**Check:** Compare with `--disable-cuda-graph` to see if TPOT changes. If TPOT stays the same, CUDA graphs weren't helping. If TPOT gets worse, CUDA graphs were partially working.

---

## Summary: Investigation Priority

| Priority | Hypothesis | Expected nsys Signal |
|----------|-----------|---------------------|
| **P0** | Scheduler prefill blocking decode | Large gaps between decode kernels; prefill kernels interleaved |
| **P1** | movedim+contiguous memcpy | D2D memcpy kernels scaling with batch size |
| **P2** | NCCL collective scaling | all_gather/reduce_scatter duration scaling super-linearly |
| **P3** | torch.empty allocation | cudaMalloc calls or allocator overhead in gaps |
| **P4** | Symmetric memory sync | Stream sync events before/after NCCL |
| **P5** | CUDA graph fallback | Missing graph replay; all ops in eager mode |

---

## Verification

1. nsys produces `.nsys-rep` files for both conditions
2. `gputrc2graph.py` generates comparison HTML showing kernel category breakdown
3. `nsys stats` output shows per-kernel timing for NCCL collectives
4. Clear identification of which component (NCCL, Triton, memcpy, allocation) dominates at c=8 vs c=1
