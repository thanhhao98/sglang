"""K3 MNNVL fused all-reduce (bf16): zero-copy AR and AR+RMSNorm.

Four entry points over the CustomAllReduceV2 storage plane (see
``csrc/kimi_k3/comm/ar_fusion.cuh``):

* :func:`all_reduce_push_res` — 1shot multicast-push. Works on ANY contiguous
  bf16 tensor (input is read and written in place); reuses the v2 push
  workspace, so the caller passes the workspace slab's multicast base.
* :func:`all_reduce_pull_res` — 2shot NVLS in place ON the input, which must
  be allocated from multicast-bound symmetric memory; the caller passes the
  input's multicast VA. The optional residual must be identical on every
  rank.
* :func:`all_reduce_push_norm` / :func:`all_reduce_pull_norm` — same two
  algorithms with a fused RMSNorm epilogue over the latent of the K3
  latent|shared MoE buffer. The row layout (num_tokens, norm rows, block /
  cluster / grid sizes) is derived and hardcoded C++-side.

All fuse ``out = allreduce(x) [+ residual/norm]`` and return ``x`` (in-place).
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
def _jit_module(world_size: int) -> Module:
    args = make_cpp_args(world_size, is_arch_support_pdl())
    cls = f"AllReduceFusionKernel<{args}>"
    return load_jit(
        "kimi_k3_all_reduce",
        *args,
        cuda_files=["kimi_k3/comm/ar_fusion.cuh"],
        cuda_wrappers=[
            ("push_res", f"{cls}::push_res"),
            ("pull_res", f"{cls}::pull_res"),
            ("push_norm", f"{cls}::push_norm"),
            ("pull_norm", f"{cls}::pull_norm"),
        ],
        extra_cuda_cflags=["-O3"],
    )


_COMM_MAP: dict[int, Communicator] = {}


def register_comm(comm: Communicator) -> None:
    """Register the CustomAllReduceV2 storage plane used by all kernels."""
    _COMM_MAP[comm.world_size] = comm


@register_custom_op(mutates_args=["x"])
def _all_reduce_push_res_op(
    world_size: int,
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    ws_mc_base: int,
) -> None:
    comm = _COMM_MAP[world_size]
    _jit_module(world_size).push_res(comm, x.view(-1), residual, ws_mc_base)


@register_custom_op(mutates_args=["x"])
def _all_reduce_pull_res_op(
    world_size: int,
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    input_mc_ptr: int,
) -> None:
    comm = _COMM_MAP[world_size]
    _jit_module(world_size).pull_res(comm, x.view(-1), residual, input_mc_ptr)


@register_custom_op(mutates_args=["x"])
def _all_reduce_push_norm_op(
    world_size: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    ws_mc_base: int,
) -> None:
    comm = _COMM_MAP[world_size]
    _jit_module(world_size).push_norm(comm, x.view(-1), weight, eps, ws_mc_base)


@register_custom_op(mutates_args=["x"])
def _all_reduce_pull_norm_op(
    world_size: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    input_mc_ptr: int,
) -> None:
    comm = _COMM_MAP[world_size]
    _jit_module(world_size).pull_norm(comm, x.view(-1), weight, eps, input_mc_ptr)


def all_reduce_push_res(
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
    _all_reduce_push_res_op(world_size, x, residual_, ws_mc_base)
    return x


def all_reduce_pull_res(
    world_size: int,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    *,
    input_mc_ptr: int,
) -> torch.Tensor:
    """In-place ``x = allreduce(x) [+ residual]`` via 2shot NVLS on ``x``.

    ``x`` MUST be allocated from multicast-bound symmetric memory and
    ``input_mc_ptr`` must be its multicast VA. The residual (if any) must be
    identical on every rank.
    """
    residual_ = residual.view(-1) if residual is not None else None
    _all_reduce_pull_res_op(world_size, x, residual_, input_mc_ptr)
    return x


def all_reduce_push_norm(
    world_size: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    ws_mc_base: int,
) -> torch.Tensor:
    """In-place allreduce + RMSNorm over the latent of the K3 latent|shared
    MoE buffer (``x`` = [N, 3584] latent then [N, 7168] shared, contiguous),
    via 1shot multicast push. N and the normed row range are derived C++-side
    from ``x``'s element count (must be a multiple of 3 * 3584)."""
    _all_reduce_push_norm_op(world_size, x, weight, eps, ws_mc_base)
    return x


def all_reduce_pull_norm(
    world_size: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    input_mc_ptr: int,
) -> torch.Tensor:
    """In-place allreduce + RMSNorm (same buffer contract as
    :func:`all_reduce_push_norm`) via NVLS 2shot; ``x`` must live in
    multicast-bound symmetric memory."""
    _all_reduce_pull_norm_op(world_size, x, weight, eps, input_mc_ptr)
    return x
