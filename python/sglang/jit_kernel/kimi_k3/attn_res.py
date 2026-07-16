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
