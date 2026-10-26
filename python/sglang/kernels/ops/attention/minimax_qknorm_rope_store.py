"""MiniMax-M3.1's grouped q/k/index norm + RoPE with the FP8 q epilogue and its NVFP4
K/V/index-K page store, in one launch.

``minimax_qknorm_rope_store_fp8`` is :func:`minimax_qknorm_rope_grouped_fp8` followed by
:func:`nvfp4_store_kv_index` on the heads it normed, bit for bit: its warps run the
q/k norm kernel's code unchanged, and the warps of the main-K and index-K heads then
store those heads (and V beside main K) with the store kernel's quantizer and
addressing (``csrc/minimax/fused_qknorm_rope_store.cuh``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.attention.minimax_qknorm_rope import _pad_groups

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# The q/k norm kernel's specialization (fused_gemma_qknorm_rope.cuh), which the
# store's caches share
HEAD_DIM, ROPE_DIM = 128, 64


@cache_once
def _jit_module(pos_dtype: torch.dtype) -> Module:
    args = make_cpp_args(pos_dtype, HEAD_DIM, ROPE_DIM, is_arch_support_pdl())
    return load_jit(
        "minimax_qknorm_rope_store",
        *args,
        cuda_files=["minimax/fused_qknorm_rope_store.cuh"],
        cuda_wrappers=[
            (
                "qknorm_rope_store_fp8",
                f"fused_gemma_qknorm_rope_store_fp8<{args}>",
            )
        ],
    )


def minimax_qknorm_rope_store_fp8(
    qkv: torch.Tensor,
    groups: Sequence[Tuple[torch.Tensor, int, int]],
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    *,
    fp8_a: torch.Tensor,
    group_a: int,
    fp8_b: torch.Tensor,
    group_b: int,
    k_group: int,
    index_k_group: int,
    k: torch.Tensor,
    v: torch.Tensor,
    idx_k: torch.Tensor,
    k_payload: torch.Tensor,
    k_scale: torch.Tensor,
    v_payload: torch.Tensor,
    v_scale: torch.Tensor,
    idx_k_payload: torch.Tensor,
    idx_k_scale: torch.Tensor,
    indices: torch.Tensor,
    global_scale: torch.Tensor,
    page_size: int,
) -> None:
    """``minimax_qknorm_rope_grouped_fp8(qkv, groups, cos_sin_cache, positions, eps,
    fp8_a=..., group_a=..., fp8_b=..., group_b=...)`` then ``nvfp4_store_kv_index(k,
    v, idx_k, k_payload, ..., indices, global_scale, page_size=page_size)``, in one
    launch, writing what the two write: ``qkv`` in place, ``fp8_a`` / ``fp8_b`` and
    the six cache buffers at ``indices``.

    ``k`` ``[T, kv_heads, 128]`` and ``idx_k`` ``[T, 128]`` must be the heads of groups
    ``k_group`` and ``index_k_group`` in place in ``qkv`` (the store reads what the norm
    wrote there), and ``v`` (shaped as ``k``) whole heads of ``qkv``'s rows outside
    every group; the kernel checks both.
    """
    num_groups, weights, offsets, counts, slot_of = _pad_groups(groups)
    slot_a, slot_b, slot_k, slot_ik = (
        slot_of[g] for g in (group_a, group_b, k_group, index_k_group)
    )
    assert min(slot_a, slot_b, slot_k, slot_ik) >= 0, "an output names an empty group"
    _jit_module(positions.dtype).qknorm_rope_store_fp8(
        qkv,
        weights[0],
        weights[1],
        weights[2],
        weights[3],
        cos_sin_cache,
        positions,
        fp8_a,
        fp8_b,
        offsets[0],
        counts[0],
        offsets[1],
        counts[1],
        offsets[2],
        counts[2],
        offsets[3],
        counts[3],
        num_groups,
        slot_a,
        slot_b,
        eps,
        slot_k,
        slot_ik,
        k,
        v,
        idx_k,
        k_payload,
        k_scale,
        v_payload,
        v_scale,
        idx_k_payload,
        idx_k_scale,
        indices,
        global_scale,
        page_size,
    )
