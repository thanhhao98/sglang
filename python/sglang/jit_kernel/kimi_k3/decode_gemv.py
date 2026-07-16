"""CUDA JIT wrapper for the tiny bf16 GEMV used by skinny K3 decode projections."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_THREADS: int = 128


@cache_once
def _jit_decode_gemv_module() -> Module:
    """Compile and cache the tiny GEMV kernel."""
    args = make_cpp_args(_THREADS, is_arch_support_pdl())
    return load_jit(
        "kimi_k3_decode_gemv_" + str(_THREADS),
        *args,
        cuda_files=["kimi_k3/decode_gemv.cuh"],
        cuda_wrappers=[("run", f"DecodeGemvKernel<{args}>::run")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


def decode_gemv(
    x: torch.Tensor,
    w: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """out[T, N] = x[T, K] @ w[N, K]^T for skinny decode projections.

    One CTA per output element; beats the cublas gemvx/dot dispatch by ~3x
    at K3 decode shapes ([7168 -> 144], [128 -> 1536]).

    x may be a row-sliced view (stride(0) != K) as long as rows are 16-byte
    aligned; the last dim must be contiguous.
    """
    if out is None:
        out = torch.empty((x.shape[0], w.shape[0]), dtype=x.dtype, device=x.device)
    _jit_decode_gemv_module().run(x, w, out)
    return out
