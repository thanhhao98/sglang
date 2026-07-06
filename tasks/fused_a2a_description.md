# Fused A2A: Single NCCL Call for Output + LSE

## What Changed

Previously, DCP A2A made **2 `all_to_all_single` calls per layer** — one for the attention output (bf16) and one for the LSE (fp32). With 27 layers, that's 54 NCCL groups per decode step, each producing `ncclDevKernel_SendRecv` GPU kernels.

The optimization **packs both into a single buffer** and does **1 call per layer** (27 total), cutting NCCL kernel count in half.

## How It Works

### Packing (before NCCL call)

```
Output: [N, B, H_per_rank, D]     bf16   (e.g., D=128)
LSE:    [N, B, H_per_rank]        fp32

1 fp32 = 2 bf16 elements (same 4 bytes, reinterpreted)

Combined buffer: [N, B, H_per_rank, D + 2]  bf16
                  ├── [:D]  = output (bf16)
                  └── [D:]  = LSE bytes viewed as 2×bf16
```

The fp32 LSE values are **not cast** to bf16 — they're **byte-reinterpreted** (`view(torch.bfloat16)`), preserving full fp32 precision. Each fp32 LSE element becomes 2 bf16-sized slots in the buffer.

### NCCL Transfer

One `all_to_all_single` on the combined flat buffer. Each rank sends/receives `B × H_per_rank × (D+2)` bf16 elements per peer — slightly more data than before (D+2 vs D), but one NCCL group instead of two.

### Unpacking (after NCCL call)

```python
recv_output = recv_combined[:, :B, :, :D]        # bf16, non-contiguous view (Triton handles strides)
recv_lse = recv_combined[:, :B, :, D:].view(fp32) # byte-reinterpret back to fp32
```

The output slice is passed directly to the Triton combine kernel as a non-contiguous view (the kernel uses explicit strides). The LSE is copied to a staging buffer for the fp32 reinterpretation (required because `view(dtype)` needs contiguous input).

## Buffer Layout (CUDA Graph)

```python
buffers = {
    "send_combined": [N, bs, H_per_rank, D + 2],  # bf16, fused send buffer
    "recv_combined": [N, bs, H_per_rank, D + 2],  # bf16, fused recv buffer
    "send_lse":      [N, bs, H_per_rank],          # fp32, staging for pack
    "recv_lse":      [N, bs, H_per_rank],          # fp32, staging for unpack
}
```

Buffers are allocated per batch size (`_alloc_dcp_a2a_buffers_for_bs`) so NCCL transfers only the actual data needed.

## Profiling Confirmation (H100, 128K, TP-0)

| Metric | Before (2 calls) | After (1 fused) |
|--------|:-:|:-:|
| SendRecv kernel count | 1512 | **756** |
| SendRecv total GPU time | 8.58ms | **4.72ms** |
| SendRecv avg per kernel | 5.7μs | 6.2μs |

Kernel count halved, total NCCL time halved. Avg per kernel slightly higher (6.2 vs 5.7μs) due to marginally larger transfer, but net savings are ~3.9ms per profiled window (~0.5ms per decode step).

## Performance Impact

Best improvement at **H100 128K c=4**: TPOT 17.20ms (was 21.82ms without fuse, 22.87ms AG_RS baseline).

At c=1, the savings are masked by NCCL minimum latency floor (~6μs per call regardless of size).
