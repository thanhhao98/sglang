"""CUDA JIT wrappers for AttnRes score and combine kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_BLOCK_H: int = 1024  # H-chunk size for score (thread count) and combine (chunk width)
_MAX_ROWS: int = 16  # next_pow2(8+1), K3 has <=8 snapshots

_DIM: int = 7168  # K3 hidden size, template parameter of the fused/chain kernels
_MAX_BANK_ROWS: int = 8  # K3 has <= 8 snapshots, upper bound of the nvb dispatch tables


def _make_name(*args):
    return "kimi_k3_attn_res_" + "_".join(str(a) for a in args)


@cache_once
def _jit_score_module() -> Module:
    """Compile and cache the JIT AttnRes score kernel."""
    args = make_cpp_args(_BLOCK_H, is_arch_support_pdl())
    return load_jit(
        _make_name("score"),
        *args,
        cuda_files=["kimi_k3/attn_res_score.cuh"],
        cuda_wrappers=[("run", f"AttnResScoreKernel<{args}>::run")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


@cache_once
def _jit_combine_module() -> Module:
    """Compile and cache the JIT AttnRes combine kernel."""
    args = make_cpp_args(_BLOCK_H, _MAX_ROWS, is_arch_support_pdl())
    return load_jit(
        _make_name("combine"),
        *args,
        cuda_files=["kimi_k3/attn_res_combine.cuh"],
        cuda_wrappers=[("run", f"AttnResCombineKernel<{args}>::run")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


@cache_once
def _jit_fused_module() -> Module:
    """Compile and cache the JIT single-kernel AttnRes aggregation."""
    args = make_cpp_args(_DIM, _MAX_BANK_ROWS, is_arch_support_pdl())
    return load_jit(
        _make_name("fused"),
        *args,
        cuda_files=["kimi_k3/attn_res/fused_reg.cuh"],
        cuda_wrappers=[("run", f"AttnResFusedKernel<{args}>::run")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


@cache_once
def _jit_chain_module() -> Module:
    """Compile and cache the optimized score/merge/norm chain kernels."""
    args = make_cpp_args(_DIM, _MAX_BANK_ROWS, is_arch_support_pdl())
    return load_jit(
        _make_name("chain"),
        *args,
        cuda_files=["kimi_k3/attn_res/chain_reg.cuh"],
        cuda_wrappers=[("run", f"AttnResChain<{args}>::run")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


@cache_once
def _jit_score_fused_add_module() -> Module:
    """Compile and cache the JIT AttnRes score+residual-add kernel."""
    args = make_cpp_args(_BLOCK_H, is_arch_support_pdl())
    return load_jit(
        _make_name("score_fused_add"),
        *args,
        cuda_files=["kimi_k3/attn_res_score_fused_add.cuh"],
        cuda_wrappers=[("run", f"AttnResScoreFusedAddKernel<{args}>::run")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


def attn_res_fused(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: torch.Tensor,
    out: torch.Tensor,
    nvb: int,
    eps: float,
) -> None:
    """Single-kernel AttnRes aggregation: per-row score -> softmax -> weighted
    combine -> output RMSNorm, one CTA per token. The bank-row count is a
    compile-time template parameter dispatched through a constexpr kernel
    table on nvb.

    Restrictions: H == 7168, SM100+ (fma.rn.f32.bf16).

    Parameters
    ----------
    prefix_sum : [T, H] bf16
    bank       : [T, NB, H] bf16 (rows 0..nvb-1 are aggregated)
    cw         : [H] bf16 — precomputed score_norm_weight * proj_weight
    ow         : [H] bf16 — output RMSNorm weight
    out        : [T, H] bf16 output buffer
    nvb        : number of valid bank rows (1..8)
    eps        : RMSNorm epsilon (shared by score and output norms)
    """
    if torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("attn_res_fused requires SM100+ (fma.rn.f32.bf16)")
    _jit_fused_module().run(prefix_sum, bank, cw, ow, out, nvb, eps)


def attn_res_chain(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: torch.Tensor,
    out: torch.Tensor,
    nvb: int,
    eps: float,
) -> None:
    """Optimized aggregation chain, one host call launching three kernels:

    - score: per-row dot(v, cw) * rrms(v), widest vectors x unroll 2; only
      the prefix-row CTA PDL-waits on the previous kernel.
    - merge: softmax(scores) -> weighted row sum, row count dispatched
      through a compile-time kernel table on nvb; rows prefetched before the
      PDL wait. Also writes per-H-chunk sum(mixed^2) partials.
    - norm: mixed * rsqrt(sum(partials)/H + eps) * ow — no reduction at all.

    The scores / partials / mixed workspace is allocated C++-side in a single
    allocation. Restrictions: H == 7168, nvb in [1, 8].

    Parameters
    ----------
    prefix_sum : [T, H] bf16
    bank       : [T, NB, H] bf16 (rows 0..nvb-1 are aggregated)
    cw         : [H] bf16 — precomputed score_norm_weight * proj_weight
    ow         : [H] bf16 — output RMSNorm weight
    out        : [T, H] bf16 output buffer
    nvb        : number of valid bank rows (1..8)
    eps        : RMSNorm epsilon (shared by score and output norms)
    """
    _jit_chain_module().run(prefix_sum, bank, cw, ow, out, nvb, eps)


def attn_res_score_fused_add(
    prefix_a: torch.Tensor,
    prefix_b: torch.Tensor,
    prefix_out: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    scores: torch.Tensor,
    nvb: int,
    eps: float,
) -> None:
    """attn_res_score with the upstream residual add fused: the prefix row is
    computed as bf16(prefix_a + prefix_b) on the fly, written to prefix_out
    (bit-identical to the standalone add), and scored."""
    _jit_score_fused_add_module().run(
        prefix_a, prefix_b, prefix_out, bank, cw, scores, nvb, eps
    )


def attn_res_score(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    scores: torch.Tensor,
    nvb: int,
    eps: float,
) -> None:
    """Launch the JIT AttnRes score kernel.

    Parameters
    ----------
    prefix_sum : [T, H] bf16
    bank       : [T, NB, H] bf16
    cw         : [H] fp32 — precomputed norm_weight * proj_weight
    scores     : [T, MAX_ROWS] fp32 output buffer
    nvb        : number of valid bank rows (0..8)
    eps        : RMSNorm epsilon
    """
    module = _jit_score_module()
    module.run(prefix_sum, bank, cw, scores, nvb, eps)


def attn_res_combine(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    scores: torch.Tensor,
    out: torch.Tensor,
    nvb: int,
) -> None:
    """Launch the JIT AttnRes combine kernel.

    Parameters
    ----------
    prefix_sum : [T, H] bf16
    bank       : [T, NB, H] bf16
    scores     : [T, MAX_ROWS] fp32 — output from attn_res_score
    out        : [T, H] bf16 output buffer
    nvb        : number of valid bank rows (0..8)
    """
    module = _jit_combine_module()
    module.run(prefix_sum, bank, scores, out, nvb)
