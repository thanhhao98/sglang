"""CUDA JIT K3 MoE-tail residual add: out = bf16(bf16(a+b) + c)."""

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

_THREADS: int = 256


@cache_once
def _jit_moe_tail_add_module() -> Module:
    args = make_cpp_args(_THREADS, is_arch_support_pdl())
    return load_jit(
        "kimi_k3_moe_tail_add_" + str(_THREADS),
        *args,
        cuda_files=["kimi_k3/moe_tail_add.cuh"],
        cuda_wrappers=[("run", f"MoeTailAddKernel<{args}>::run")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


def kimi_k3_moe_tail_add(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """out = bf16(bf16(a + b) + c); double rounding matches the unfused pair
    bit-for-bit. b may be a row-sliced view (contiguous last dim)."""
    out = torch.empty_like(a)
    _jit_moe_tail_add_module().run(a, b, c, out)
    return out
