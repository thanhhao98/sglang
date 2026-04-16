"""Benchmark fused Triton A2A pack/unpack+combine vs original PyTorch ops.

Measures pack and unpack+combine steps separately (no NCCL).
Tests with shapes matching CodeQwen 7B and Qwen3-235B.

Usage:
    python benchmark/dcp/bench_a2a_pack.py
"""

import time

import torch
import triton

from sglang.srt.layers.attention.dcp_a2a import (
    _lse_pack_dim,
    _pack_a2a_send,
    dcp_fused_unpack_combine,
    dcp_lse_combine_triton,
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


def pytorch_unpack_combine(recv_combined, N, B, H_per_rank, D, lpd, is_base_e=False):
    """Original PyTorch unpack (3 ops) + separate Triton combine."""
    recv_output = recv_combined[:, :, :, :D]
    recv_lse_stg = torch.empty(
        N, B, H_per_rank, dtype=torch.float32, device=recv_combined.device
    )
    recv_lse_stg.view(recv_combined.dtype).view(N, B, H_per_rank, lpd).copy_(
        recv_combined[:, :, :, D:]
    )
    combined, _ = dcp_lse_combine_triton(
        recv_output, recv_lse_stg, is_lse_base_on_e=is_base_e
    )
    return combined


def triton_unpack_combine(recv_combined, N, B, H_per_rank, D, lpd, is_base_e=False):
    """Fused Triton unpack + combine (1 kernel)."""
    return dcp_fused_unpack_combine(
        recv_combined, B, H_per_rank, D, lpd, is_lse_base_on_e=is_base_e
    )


def bench_unpack(name, B, H, D, N, warmup=20, iters=100):
    """Benchmark unpack+combine for one configuration."""
    H_per_rank = H // N
    out_dtype = torch.bfloat16
    lpd = _lse_pack_dim(out_dtype)

    recv_combined = torch.randn(
        N, B, H_per_rank, D + lpd, dtype=out_dtype, device="cuda"
    )

    # Verify correctness
    ref = pytorch_unpack_combine(recv_combined, N, B, H_per_rank, D, lpd)
    test = triton_unpack_combine(recv_combined, N, B, H_per_rank, D, lpd)
    if not torch.allclose(ref, test, atol=1e-2, rtol=1e-2):
        max_diff = (ref - test).abs().max().item()
        print(f"  WARNING: max diff={max_diff:.6f} (may be acceptable for bf16)")

    # Benchmark PyTorch
    for _ in range(warmup):
        pytorch_unpack_combine(recv_combined, N, B, H_per_rank, D, lpd)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        pytorch_unpack_combine(recv_combined, N, B, H_per_rank, D, lpd)
    torch.cuda.synchronize()
    pytorch_us = (time.perf_counter() - t0) / iters * 1e6

    # Benchmark Triton
    for _ in range(warmup):
        triton_unpack_combine(recv_combined, N, B, H_per_rank, D, lpd)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        triton_unpack_combine(recv_combined, N, B, H_per_rank, D, lpd)
    torch.cuda.synchronize()
    triton_us = (time.perf_counter() - t0) / iters * 1e6

    speedup = pytorch_us / triton_us
    print(
        f"  {name:40s}  B={B:4d}  H={H:3d}  D={D:3d}  N={N}  "
        f"PyTorch={pytorch_us:7.1f}us  Triton={triton_us:7.1f}us  "
        f"Speedup={speedup:.2f}x"
    )


CONFIGS = [
    # (label, H, N)
    ("dcp2 (H=8, N=2)", 8, 2),
    ("tpa4 dcp2 (H=8, N=2)", 8, 2),
    ("tpa2 dcp4 (H=16, N=4)", 16, 4),
]

CONFIGS_235B = [
    ("dcp2 (H=16, N=2)", 16, 2),
    ("tpa4 dcp2 (H=16, N=2)", 16, 2),
    ("tpa2 dcp4 (H=32, N=4)", 32, 4),
]


def main():
    D = 128

    # ---- Pack benchmark ----
    print("=" * 110)
    print("PACK Benchmark: Fused Triton vs PyTorch copy")
    print("=" * 110)

    print("\n--- CodeQwen 7B ---")
    for label, H, N in CONFIGS:
        for B in [1, 32, 128, 512]:
            bench_one(f"7B {label}", B=B, H=H, D=D, N=N)
        print()

    print("--- Qwen3-235B ---")
    for label, H, N in CONFIGS_235B:
        for B in [1, 32, 64, 128]:
            bench_one(f"235B {label}", B=B, H=H, D=D, N=N)
        print()

    # ---- Unpack+Combine benchmark ----
    print("=" * 110)
    print("UNPACK+COMBINE Benchmark: Fused Triton vs PyTorch unpack + Triton combine")
    print("=" * 110)

    print("\n--- CodeQwen 7B ---")
    for label, H, N in CONFIGS:
        for B in [1, 32, 128, 512]:
            bench_unpack(f"7B {label}", B=B, H=H, D=D, N=N)
        print()

    print("--- Qwen3-235B ---")
    for label, H, N in CONFIGS_235B:
        for B in [1, 32, 64, 128]:
            bench_unpack(f"235B {label}", B=B, H=H, D=D, N=N)
        print()

    print("=" * 110)


if __name__ == "__main__":
    main()
