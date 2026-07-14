"""Native-CUDA radix-select router for the K3 decode regime (kda-pilot task 08).

Replaces the triton router's 16 dependent argmax rounds with byte-histogram radix
narrowing (single CTA per token). Selection semantics match _router_triton_kernel
exactly (ids bit-identical incl. adversarial ties/NaN; weights <= 2.4e-7 rel).
Standalone-validated: 1.79x over the tuned in-tree triton router (graphed A/B on
GB300), 1000-replay stable, CUDA-graph capturable.

Covered regime (caller guards + `covered()`): scores [M,896] bf16 row-contiguous,
bias [896] fp32, topk=16. JIT-built via torch cpp_extension on first use.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT = None


def covered(scores: torch.Tensor, bias: torch.Tensor, topk: int) -> bool:
    return (
        scores.dim() == 2
        and scores.size(1) == 896
        and int(topk) == 16
        and scores.dtype == torch.bfloat16
        and bias.dtype == torch.float32
        and scores.stride(1) == 1
        and bias.is_contiguous()
    )


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        _EXT = load(
            name="bs1_route_radix",
            sources=[os.path.join(_HERE, "route_radix.cu")],
            extra_cuda_cflags=["-O3", "-lineinfo"],  # no one-sided fast-math
            verbose=False,
        )
    return _EXT


def route_radix(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_scale: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (weights [M,topk] fp32, ids [M,topk] int32). Caller must have
    checked covered()."""
    M = scores.shape[0]
    out_w = torch.empty((M, topk), dtype=torch.float32, device=scores.device)
    out_i = torch.empty((M, topk), dtype=torch.int32, device=scores.device)
    _ext().run(
        scores,
        bias,
        out_w,
        out_i,
        int(topk),
        float(routed_scaling_factor),
        bool(renormalize),
        bool(apply_scale),
    )
    return out_w, out_i
