"""K3 MNNVL fused all-reduce (bf16): zero-copy AR and AR+residual.

Two entry points over the CustomAllReduceV2 storage plane (see
``csrc/kimi_k3/comm/ar_fusion.cuh``):

* :func:`ar_fusion_push` — 1shot multicast-push. Works on ANY contiguous
  bf16 tensor (input is read and written in place); reuses the v2 push
  workspace, so the caller passes the workspace slab's multicast base.
* :func:`ar_fusion_pull_mc` — 2shot NVLS in place ON the input, which must
  be allocated from multicast-bound symmetric memory; the caller passes the
  input's multicast VA. The optional residual must be identical on every
  rank.

Both fuse ``out = allreduce(x) [+ residual]`` and return ``x`` (in-place).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

    from sglang.jit_kernel.all_reduce import Communicator


@cache_once
def _jit_ar_fusion_module(world_size: int) -> Module:
    args = make_cpp_args(world_size, is_arch_support_pdl())
    cls = f"ArFusionKernel<{args}>"
    return load_jit(
        "kimi_k3_ar_fusion",
        *args,
        cuda_files=["kimi_k3/comm/ar_fusion.cuh"],
        cuda_wrappers=[
            ("push", f"{cls}::push"),
            ("pull_mc", f"{cls}::pull_mc"),
        ],
        extra_cuda_cflags=["-O3"],
    )


# Communicator instances are not traceable through custom ops; register the
# storage plane once per world size and route through the map (same trick as
# MiniMaxM2QKRMSNorm.COMM_MAP).
_COMM_MAP: dict[int, Communicator] = {}


@register_custom_op(mutates_args=["x"])
def _ar_fusion_push_op(
    world_size: int,
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    ws_mc_base: int,
) -> None:
    comm = _COMM_MAP[world_size]
    module = _jit_ar_fusion_module(world_size)
    module.push(comm, x.view(-1), residual, ws_mc_base)


@register_custom_op(mutates_args=["x"])
def _ar_fusion_pull_mc_op(
    world_size: int,
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    input_mc_ptr: int,
    num_blocks: int,
) -> None:
    comm = _COMM_MAP[world_size]
    module = _jit_ar_fusion_module(world_size)
    module.pull_mc(comm, x.view(-1), residual, input_mc_ptr, num_blocks)


def register_comm(comm: Communicator) -> None:
    """Register the CustomAllReduceV2 storage plane used by both kernels."""
    _COMM_MAP[comm.world_size] = comm


def ar_fusion_push(
    world_size: int,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    *,
    ws_mc_base: int,
) -> torch.Tensor:
    """In-place ``x = allreduce(x) [+ residual]`` via 1shot multicast push.

    ``x`` may be any contiguous bf16 CUDA tensor whose byte size fits the
    registered push workspace. ``ws_mc_base`` is the multicast VA of the v2
    workspace slab base. Call :func:`register_comm` once beforehand.
    """
    residual_ = residual.view(-1) if residual is not None else None
    _ar_fusion_push_op(world_size, x, residual_, ws_mc_base)
    return x


def ar_fusion_pull_mc(
    world_size: int,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    *,
    input_mc_ptr: int,
    num_blocks: int = 0,
) -> torch.Tensor:
    """In-place ``x = allreduce(x) [+ residual]`` via 2shot NVLS on ``x``.

    ``x`` MUST be allocated from multicast-bound symmetric memory and
    ``input_mc_ptr`` must be its multicast VA. The residual (if any) must be
    identical on every rank. ``num_blocks == 0`` picks the tuned default.
    """
    residual_ = residual.view(-1) if residual is not None else None
    _ar_fusion_pull_mc_op(world_size, x, residual_, input_mc_ptr, num_blocks)
    return x
