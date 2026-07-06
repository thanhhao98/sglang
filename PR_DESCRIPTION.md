# [Feature] DCP: Decode Context Parallelism with A2A and FA3 Backend Support

## Motivation

This PR extends the DCP (Decode Context Parallelism) feature from [#14194](https://github.com/sgl-project/sglang/pull/14194) by @staugust with two major additions:

1. **A2A (All-to-All) communication backend** -- an alternative to AllGather+ReduceScatter that reduces NCCL calls per layer from 2 to 1 by fusing output+LSE into a single exchange.
2. **FA3 (FlashAttention-3) backend support** -- enables DCP with the FA3 attention backend, not just FlashInfer.

These extensions make DCP production-ready with multiple communication strategies and attention backends, giving users flexibility to choose the best combination for their hardware.

## Background: What is DCP?

DCP splits the KV cache across ranks within a TP group during decode. Each rank stores only `1/dcp_size` of the KV cache tokens (interleaved by position), computes partial attention over its local shard, then combines results using LSE-weighted merging to produce the correct full-attention output.

This allows serving much longer contexts (e.g., 256K-1M tokens) on the same hardware by distributing KV cache memory across GPUs, at the cost of additional communication during decode.

The base DCP implementation in [#14194](https://github.com/sgl-project/sglang/pull/14194) supports FlashInfer with AllGather+ReduceScatter (AG+RS) communication. This PR adds:

## Modifications

### 1. A2A Communication Backend (`dcp_a2a.py`)

**New file: `python/sglang/srt/layers/attention/dcp_a2a.py`**

Instead of AllGather(Q) -> Attention -> AllGather(output) -> LSE-correct -> ReduceScatter(output), the A2A backend:
- Runs attention with all heads on local KV shard (heads already distributed by TP)
- Packs output + LSE into a fused buffer (`[N, B, H_per_rank, D + lse_pack_dim]`)
- Executes a single `all_to_all_single` to exchange head partials between ranks
- Combines received partials with a Triton LSE-weighted combine kernel

This halves NCCL collective calls per MLA layer (from 2 to 1), reducing communication overhead.

Key components:
- `dcp_lse_combine_triton`: Triton kernel for LSE-weighted output combination (supports both base-e FA3 and base-2 FlashInfer LSE conventions)
- `dcp_a2a_lse_reduce`: Fused A2A exchange + local combine with CUDA graph buffer support
- `_lse_weighted_combine_cpu`: CPU reference implementation for testing
- `PyNcclCommunicator.all_to_all_single`: NCCL-based A2A using ncclGroupStart/End for graph-capturability

### 2. FA3 Backend Support (`flashattention_backend.py`)

Extended the FlashAttention backend to handle DCP decode and extend paths:
- DCP metadata computation (head counts, group references)
- Q AllGather across DCP group for decode
- KV prefix AllGather for extend
- LSE-weighted output correction after attention (AG+RS or A2A)
- Cascade attention guard when DCP > 1
- CUDA graph buffer pre-allocation for DCP A2A

### 3. Server Args and Configuration

- `--dcp-size N`: DCP world size (replaces `SGLANG_DCP` env var)
- `--dcp-comm-backend {ag_rs, a2a}`: Communication backend choice
- Validation: A2A requires `dcp_size > 1`; `tp_size` must be divisible by `dcp_size`

### 4. Symmetric Memory Support

- Extended `pynccl_allocator.py` to support multiple symmetric memory groups (TP + DCP)
- DCP group uses symmetric memory for AllGather/ReduceScatter under CUDA graph capture
- `SGLANG_DCP_SYMM_ONLY` env var to enable symmetric memory exclusively for DCP group

### 5. CI-Registered Tests

Added 7 test files under `test/registered/` for CI auto-discovery:

| File | Suite | Tests |
|------|-------|-------|
| `dcp/test_dcp_accuracy.py` | `stage-c-test-8-gpu-h200` | 4 E2E configs (FlashInfer/FA3 x AG+RS/A2A) |
| `kernels/test_dcp_lse_combine.py` | `stage-b-test-1-gpu-large` | 21 Triton kernel correctness tests |
| `kernels/test_dcp_interleaved.py` | `stage-b-test-1-gpu-small` | 11 KV allocator interleaved storage tests |
| `kernels/test_dcp_fa3_standalone.py` | `stage-b-test-1-gpu-large` | 6 FA3 MLA + simulated DCP tests |
| `unit/server_args/test_dcp_config.py` | `stage-a-test-cpu` | 8 ServerArgs validation tests |
| `unit/layers/test_dcp_cascade_guard.py` | `stage-a-test-cpu` | 15 cascade attention guard tests |
| `unit/layers/test_dcp_need_lse.py` | `stage-a-test-cpu` | 4 need_lse logic tests |

### 6. Symmetric Memory Benchmark

**`benchmark/kernels/all_reduce/benchmark_symm_mem.py`**: Benchmarks AllGather, ReduceScatter, and All-to-All collectives comparing torch eager vs PyNccl symmetric-memory CUDA graph.

## Benchmarking and Profiling

### Serving Performance: DCP vs TP8 Baseline

Benchmarked with DeepSeek-V2 on 8x H100, using `bench_serving` with random dataset (input ~4000 tokens, output ~1500 tokens) across concurrency levels 1-512.

![Benchmark Comparison](bench_result_news/bench_comparison.png)

**Key finding: DCP enables 2.2x higher throughput at high concurrency** by distributing KV cache across GPUs, allowing the system to serve more concurrent requests before running out of memory.

#### Output Token Throughput (tok/s)

| Config | cc1 | cc8 | cc32 | cc64 | cc128 | cc256 | cc512 |
|--------|-----|-----|------|------|-------|-------|-------|
| TP8 FlashInfer (baseline) | 103 | 498 | 1060 | **1385** | 1374 | 1403 | 1400 |
| TP8 FA3 (baseline) | 103 | 492 | 1058 | **1370** | 1358 | 1384 | 1382 |
| DCP8 AG+RS FlashInfer | 89 | 425 | 947 | 1336 | 1945 | 2568 | **3107** |
| DCP8 AG+RS FA3 | 90 | 425 | 953 | 1358 | 1961 | 2572 | **2930** |
| DCP8 A2A FlashInfer | 86 | 413 | 929 | 1320 | 1919 | 2559 | **3126** |
| DCP8 A2A FA3 | 87 | 413 | 936 | 1341 | 1933 | 2570 | **2951** |

- TP8 **plateaus at ~1400 tok/s** around cc64 -- KV cache is full, no more requests can be served concurrently.
- DCP8 **continues scaling to 2900-3100 tok/s** at cc512 -- 8x more KV cache capacity from interleaved distribution.
- At cc512: DCP delivers **2.2x the throughput** of TP8.

#### Mean TTFT (ms) -- Time to First Token

| Config | cc32 | cc64 | cc128 | cc256 | cc512 |
|--------|------|------|-------|-------|-------|
| TP8 FlashInfer | 413 | **5,744** | **39,672** | **106,844** | **239,577** |
| DCP8 AG+RS FlashInfer | 420 | 655 | 1,095 | 1,970 | **28,734** |

- TP8 TTFT **explodes to 240 seconds** at cc512 due to request queuing when KV cache is full.
- DCP8 TTFT stays at **28 seconds** -- a **8.3x improvement** because more requests fit in memory.

#### Per-Token Latency (TPOT/ITL)

At low concurrency (cc1-cc32), TP8 has ~15% lower per-token latency since DCP adds communication overhead per decode step. However, TP8 cannot sustain higher concurrency at all -- it queues requests instead, making the latency comparison moot above cc64.

### Accuracy Tests

GSM8K few-shot accuracy with DeepSeek-V2 on 8x H100, TP8, DCP8 (200 questions):

| Configuration | Accuracy |
|---------------|----------|
| TP8 (baseline, no DCP) | 0.805-0.810 |
| DCP8 FlashInfer + AG+RS | 0.810 |
| DCP8 FlashInfer + A2A | 0.800 |
| DCP8 FA3 + AG+RS | 0.790 |
| DCP8 FA3 + A2A | 0.800 |

All DCP configurations match baseline TP8 accuracy within noise margin.

### Symmetric Memory Benchmark (H100 8-GPU)

| msg_size | AG eager (us) | AG symm graph (us) | RS eager (us) | RS symm graph (us) | A2A eager (us) | A2A symm graph (us) |
|----------|--------------|---------------------|---------------|---------------------|----------------|----------------------|
| 2 KiB | 14.57 | 2.68 | 16.06 | 2.82 | 18.34 | 5.45 |
| 8 KiB | 14.42 | 3.11 | 15.82 | 3.00 | 17.84 | 5.57 |
| 32 KiB | 14.37 | 5.07 | 18.74 | 3.25 | 17.90 | 6.25 |
| 128 KiB | 18.14 | 6.90 | 17.12 | 4.24 | 21.18 | 7.00 |

Symmetric memory CUDA graph speedup: AG 2.6-5.4x, RS 4.0-6.2x, A2A 3.0-3.6x.

## Usage

```bash
# DCP with AG+RS (default, compatible with CUDA graph)
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V2 \
    --tp-size 8 --dcp-size 8 \
    --dcp-comm-backend ag_rs \
    --attention-backend flashinfer \
    --enable-symm-mem --disable-radix-cache \
    --trust-remote-code

# DCP with A2A communication
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V2 \
    --tp-size 8 --dcp-size 8 \
    --dcp-comm-backend a2a \
    --attention-backend fa3 \
    --enable-symm-mem --disable-radix-cache \
    --disable-cuda-graph \
    --trust-remote-code
```

## Acknowledgment

This work builds on the DCP implementation by @staugust in [#14194](https://github.com/sgl-project/sglang/pull/14194), which introduced the core DCP infrastructure: distributed group setup, interleaved KV cache storage, FlashInfer MLA backend integration, and the LSE correction kernel. Our contributions extend this foundation with A2A communication, FA3 backend support, symmetric memory optimization, proper CLI args, and comprehensive CI-registered tests.

## Checklist

- [x] Format code with pre-commit
- [x] Add unit tests (69 tests across 7 files, all passing)
- [x] Accuracy benchmark (GSM8K: 0.79-0.81 across all configs)
- [x] Performance benchmark (symmetric memory collectives)
- [x] Follow SGLang code style
