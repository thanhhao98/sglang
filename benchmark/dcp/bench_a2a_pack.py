"""Benchmark fused Triton A2A pack vs original PyTorch copy.

Measures the pack step only (no NCCL, no combine).
Tests with shapes matching CodeQwen 7B and Qwen3-235B.

Usage:
    python benchmark/dcp/bench_a2a_pack.py
"""

import time

import torch
import triton

# Import both implementations
from sglang.srt.layers.attention.dcp_a2a import (
    _lse_pack_dim,
    _pack_a2a_send,
)


def pytorch_pack(cp_attn_out, cp_attn_lse, N, B, H_per_rank, D, lpd, out_dtype):
    """Original PyTorch copy-based pack (3 separate ops)."""
    reshaped_out = cp_attn_out.view(B, N, H_per_rank, D).permute(1, 0, 2, 3)
    reshaped_lse = cp_attn_lse.view(B, N, H_per_rank).permute(1, 0, 2)

    send_combined = torch.empty(
        N, B, H_per_rank, D + lpd, dtype=out_dtype, device=cp_attn_out.device
    )

    send_lse_contig = reshaped_lse.contiguous()
    send_combined[:, :, :, :D].copy_(reshaped_out)
    send_combined[:, :, :, D:].copy_(
        send_lse_contig.view(out_dtype).view(N, B, H_per_rank, lpd)
    )
    return send_combined


def triton_pack(cp_attn_out, cp_attn_lse, N, B, H_per_rank, D, lpd, out_dtype):
    """Fused Triton pack (1 kernel)."""
    send_combined = torch.empty(
        N, B, H_per_rank, D + lpd, dtype=out_dtype, device=cp_attn_out.device
    )
    _pack_a2a_send(cp_attn_out, cp_attn_lse, send_combined, N, B, H_per_rank, D, lpd)
    return send_combined


def bench_one(name, B, H, D, N, warmup=20, iters=100):
    """Benchmark one configuration."""
    H_per_rank = H // N
    out_dtype = torch.bfloat16
    lpd = _lse_pack_dim(out_dtype)

    cp_attn_out = torch.randn(B, H, D, dtype=out_dtype, device="cuda")
    cp_attn_lse = torch.randn(B, H, dtype=torch.float32, device="cuda")

    # Verify correctness
    ref = pytorch_pack(cp_attn_out, cp_attn_lse, N, B, H_per_rank, D, lpd, out_dtype)
    test = triton_pack(cp_attn_out, cp_attn_lse, N, B, H_per_rank, D, lpd, out_dtype)
    assert torch.allclose(ref[:, :, :, :D], test[:, :, :, :D], atol=1e-3), "Output mismatch!"

    ref_lse = ref[:, :, :, D:].contiguous().view(torch.float32).view(N, B, H_per_rank)
    test_lse = test[:, :, :, D:].contiguous().view(torch.float32).view(N, B, H_per_rank)
    assert torch.allclose(ref_lse, test_lse, atol=1e-5), "LSE mismatch!"

    # Benchmark PyTorch
    for _ in range(warmup):
        pytorch_pack(cp_attn_out, cp_attn_lse, N, B, H_per_rank, D, lpd, out_dtype)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        pytorch_pack(cp_attn_out, cp_attn_lse, N, B, H_per_rank, D, lpd, out_dtype)
    torch.cuda.synchronize()
    pytorch_us = (time.perf_counter() - t0) / iters * 1e6

    # Benchmark Triton
    for _ in range(warmup):
        triton_pack(cp_attn_out, cp_attn_lse, N, B, H_per_rank, D, lpd, out_dtype)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        triton_pack(cp_attn_out, cp_attn_lse, N, B, H_per_rank, D, lpd, out_dtype)
    torch.cuda.synchronize()
    triton_us = (time.perf_counter() - t0) / iters * 1e6

    speedup = pytorch_us / triton_us
    print(
        f"  {name:40s}  B={B:4d}  H={H:3d}  D={D:3d}  N={N}  "
        f"PyTorch={pytorch_us:7.1f}us  Triton={triton_us:7.1f}us  "
        f"Speedup={speedup:.2f}x"
    )


def main():
    print("=" * 100)
    print("A2A Pack Benchmark: Fused Triton vs PyTorch copy")
    print("=" * 100)

    # CodeQwen 7B: 32 Q heads, 4 KV heads, head_dim=128
    # attn_tp=4 → 8 heads/rank, attn_tp=2 → 16 heads/rank
    print("\n--- CodeQwen 7B (head_dim=128) ---")
    for B in [1, 32, 64, 128, 256, 512]:
        bench_one("7B tp8 dcp2 (H=8, N=2)", B=B, H=8, D=128, N=2)
    print()
    for B in [1, 32, 64, 128, 256, 512]:
        bench_one("7B tpa4 dcp2 (H=8, N=2)", B=B, H=8, D=128, N=2)
    print()
    for B in [1, 32, 64, 128, 256, 512]:
        bench_one("7B tpa2 dcp4 (H=16, N=4)", B=B, H=16, D=128, N=4)

    # Qwen3-235B: 64 Q heads, 4 KV heads, head_dim=128
    # attn_tp=4 → 16 heads/rank, attn_tp=2 → 32 heads/rank
    print("\n--- Qwen3-235B (head_dim=128) ---")
    for B in [1, 32, 64, 128]:
        bench_one("235B dcp2 (H=16, N=2)", B=B, H=16, D=128, N=2)
    print()
    for B in [1, 32, 64, 128]:
        bench_one("235B tpa4 dcp2 (H=16, N=2)", B=B, H=16, D=128, N=2)
    print()
    for B in [1, 32, 64, 128]:
        bench_one("235B tpa2 dcp4 (H=32, N=4)", B=B, H=32, D=128, N=4)

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
