# SPDX-License-Identifier: Apache-2.0
# Kimi-K3 Attention Residual aggregation kernels.
#
# Each aggregation point: score rows → softmax → weighted sum → RMSNorm.
# Four modes via SGLANG_K3_ATTN_RES_MODE:
#   "fused"  — Triton 3-kernel pipeline with full H-parallelism (default)
#   "jit"    — CUDA JIT kernels with lower launch overhead than Triton
#   "torch"  — PyTorch reference (readable, for debugging)
#   "legacy" — original path in kimi_k3.py

import os

import torch
import triton
import triton.language as tl

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear

_MODE = os.environ.get("SGLANG_K3_ATTN_RES_MODE", "fused")
_BLOCK_H: int = 1024  # H=7168 = 7 × 1024
_MAX_ROWS: int = 16  # next_pow2(8+1), K3 has ≤8 snapshots


# ---- Precomputed weight cache ------------------------------------------------


def get_cw(proj: ReplicatedLinear, norm: RMSNorm) -> torch.Tensor:
    """Cached fp32 product: norm_weight ⊙ proj_weight (both [H])."""
    cw = getattr(proj, "_attn_res_cw", None)
    if cw is None:
        cw = (norm.weight.float() * proj.weight.squeeze().float()).contiguous()
        proj._attn_res_cw = cw
    return cw


# ---- Kernel 1: per-row scoring (2D grid [T, NVB+1]) -------------------------


@triton.jit
def _score_kernel(
    prefix_ptr,  # [T, H]
    bank_ptr,  # [T, NB_total, H]
    cw_ptr,  # [H] fp32
    scores_ptr,  # [T, MAX_ROWS] fp32
    NVB,
    eps,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_sm,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One CTA per (token, row): scan H, output one scalar score."""
    pid_t = tl.program_id(0)
    j = tl.program_id(1)
    if j > NVB:
        return
    sumsq = 0.0
    dotv = 0.0
    for h0 in tl.static_range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        if j < NVB:
            v = tl.load(bank_ptr + pid_t * stride_bm + j * stride_bb + offs_h).to(
                tl.float32
            )
        else:
            v = tl.load(prefix_ptr + pid_t * stride_pm + offs_h).to(tl.float32)
        cw = tl.load(cw_ptr + offs_h)
        sumsq += tl.sum(v * v)
        dotv += tl.sum(v * cw)
    rrms = 1.0 / tl.sqrt(sumsq / H + eps)
    tl.store(scores_ptr + pid_t * stride_sm + j, dotv * rrms)


# ---- Kernel 2: softmax + weighted sum (2D grid [T, H/BLOCK_H]) --------------


@triton.jit
def _combine_kernel(
    prefix_ptr,
    bank_ptr,
    scores_ptr,  # [T, MAX_ROWS] fp32
    out_ptr,  # [T, H]
    NVB,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_sm,
    stride_om,
    BLOCK_H: tl.constexpr,
    MAX_ROWS: tl.constexpr,
):
    """One CTA per (token, H-chunk): softmax(scores) → weighted sum → write chunk.

    Softmax is redundantly computed by each H-chunk CTA (≤16 elements, trivial).
    This gives full H-parallelism: 7 CTAs for H=7168/1024.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    h0 = pid_h * BLOCK_H

    # Softmax (redundant per chunk, 16 fp32 ops)
    offs_b = tl.arange(0, MAX_ROWS)
    mask_b = offs_b <= NVB
    raw = tl.load(
        scores_ptr + pid_t * stride_sm + offs_b, mask=mask_b, other=float("-inf")
    )
    m = tl.max(raw, axis=0)
    e = tl.where(mask_b, tl.exp(raw - m), 0.0)
    p = e / tl.sum(e, axis=0)

    # Weighted sum for this H chunk
    offs_h = h0 + tl.arange(0, BLOCK_H)
    acc = tl.zeros([BLOCK_H], tl.float32)
    for j in range(0, NVB + 1):
        if j < NVB:
            v = tl.load(bank_ptr + pid_t * stride_bm + j * stride_bb + offs_h).to(
                tl.float32
            )
        else:
            v = tl.load(prefix_ptr + pid_t * stride_pm + offs_h).to(tl.float32)
        p_j = tl.sum(tl.where(offs_b == j, p, 0.0), axis=0)
        acc += p_j * v
    tl.store(
        out_ptr + pid_t * stride_om + offs_h,
        acc.to(out_ptr.dtype.element_ty),
    )


# ---- Fused path: score → combine → RMSNorm(standard) ------------------------


def _aggregate_fused(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: ReplicatedLinear,
    score_norm: RMSNorm,
    out_norm: RMSNorm,
) -> torch.Tensor:
    T, H = prefix_sum.shape
    cw = get_cw(score_proj, score_norm)
    n_h_blocks = H // _BLOCK_H

    # Step 1: score each row (2D grid, full row-parallelism)
    scores = torch.empty((T, _MAX_ROWS), dtype=torch.float32, device=prefix_sum.device)
    _score_kernel[(T, nvb + 1)](
        prefix_sum,
        bank,
        cw,
        scores,
        nvb,
        score_norm.variance_epsilon,
        prefix_sum.stride(0),
        bank.stride(0),
        bank.stride(1),
        scores.stride(0),
        H=H,
        BLOCK_H=_BLOCK_H,
        num_warps=8,
    )

    # Step 2: softmax + weighted sum (2D grid, full H-parallelism)
    out = torch.empty_like(prefix_sum)
    _combine_kernel[(T, n_h_blocks)](
        prefix_sum,
        bank,
        scores,
        out,
        nvb,
        prefix_sum.stride(0),
        bank.stride(0),
        bank.stride(1),
        scores.stride(0),
        out.stride(0),
        BLOCK_H=_BLOCK_H,
        MAX_ROWS=_MAX_ROWS,
        num_warps=4,
    )

    # Step 3: standard RMSNorm (sglang's optimized kernel)
    return out_norm(out)


# ---- JIT CUDA path: score → combine → RMSNorm(standard) ---------------------


def _aggregate_jit(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: ReplicatedLinear,
    score_norm: RMSNorm,
    out_norm: RMSNorm,
) -> torch.Tensor:
    from sglang.jit_kernel.kimi_k3.attn_res import attn_res_combine, attn_res_score

    T, H = prefix_sum.shape
    cw = get_cw(score_proj, score_norm)
    n_h_blocks = H // _BLOCK_H

    # Step 1: score each row (2D grid, full row-parallelism)
    scores = torch.empty((T, _MAX_ROWS), dtype=torch.float32, device=prefix_sum.device)
    attn_res_score(prefix_sum, bank, cw, scores, nvb, score_norm.variance_epsilon)

    # Step 2: softmax + weighted sum (2D grid, full H-parallelism)
    out = torch.empty_like(prefix_sum)
    attn_res_combine(prefix_sum, bank, scores, out, nvb)

    # Step 3: standard RMSNorm (sglang's optimized kernel)
    return out_norm(out)


# ---- PyTorch reference -------------------------------------------------------


def _aggregate_torch(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: ReplicatedLinear,
    score_norm: RMSNorm,
    out_norm: RMSNorm,
) -> torch.Tensor:
    T, H = prefix_sum.shape
    # rows = [bank[0..nvb-1], prefix_sum]  shape [T, nvb+1, H]
    rows = torch.cat([bank[:, :nvb, :], prefix_sum.unsqueeze(1)], dim=1)
    R = nvb + 1
    # score: RMSNorm each row, project to scalar
    normed = score_norm(rows.reshape(T * R, H))
    scores = score_proj(normed)[0].reshape(T, R)
    # softmax + weighted sum of original rows
    probs = torch.softmax(scores.float(), dim=-1)
    mixed = (probs.unsqueeze(-1) * rows.float()).sum(dim=1)
    # output norm
    return out_norm(mixed.to(prefix_sum.dtype))


# ---- Fused residual-add + aggregation (jit mode) -----------------------------


def attn_res_aggregate_fadd(
    prefix_a: torch.Tensor,
    prefix_b: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: ReplicatedLinear,
    score_norm: RMSNorm,
    out_norm: RMSNorm,
):
    """Aggregation point with the upstream residual add fused into the score
    kernel: prefix = bf16(prefix_a + prefix_b) is computed on the fly by the
    prefix-row CTA (bit-identical to the standalone add) and materialized for
    the combine kernel / downstream consumers. Returns (normed, prefix).

    jit mode only; other modes fall back to add-then-aggregate."""
    if _MODE == "jit":
        from sglang.jit_kernel.kimi_k3.attn_res import (
            attn_res_combine,
            attn_res_score_fadd,
        )

        T, H = prefix_a.shape
        cw = get_cw(score_proj, score_norm)
        prefix = torch.empty_like(prefix_a)
        scores = torch.empty(
            (T, _MAX_ROWS), dtype=torch.float32, device=prefix_a.device
        )
        attn_res_score_fadd(
            prefix_a,
            prefix_b,
            prefix,
            bank,
            cw,
            scores,
            nvb,
            score_norm.variance_epsilon,
        )
        out = torch.empty_like(prefix)
        attn_res_combine(prefix, bank, scores, out, nvb)
        return out_norm(out), prefix

    prefix = prefix_a + prefix_b
    return (
        attn_res_aggregate(prefix, bank, nvb, score_proj, score_norm, out_norm),
        prefix,
    )


# ---- Public dispatch ---------------------------------------------------------


def attn_res_aggregate(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: ReplicatedLinear,
    score_norm: RMSNorm,
    out_norm: RMSNorm,
) -> torch.Tensor:
    """Single aggregation point: score → softmax → mix → norm.

    Caller handles nvb == 0 (layer 0 attn side: just out_norm(prefix_sum)).
    """
    if _MODE == "torch":
        return _aggregate_torch(prefix_sum, bank, nvb, score_proj, score_norm, out_norm)
    if _MODE == "jit":
        return _aggregate_jit(prefix_sum, bank, nvb, score_proj, score_norm, out_norm)
    return _aggregate_fused(prefix_sum, bank, nvb, score_proj, score_norm, out_norm)
