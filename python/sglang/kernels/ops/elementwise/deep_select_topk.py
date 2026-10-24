from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import KERNEL_PATH, cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

DEEP_SELECT_INPUT_STRIDE_ALIGNMENT = 256
_DEEP_SELECT_SUPPORTED_K = (512,)


def _check_topk(topk: int) -> None:
    if topk not in _DEEP_SELECT_SUPPORTED_K:
        raise RuntimeError(
            f"Unsupported topk {topk}. Supported: {_DEEP_SELECT_SUPPORTED_K}"
        )


def _deep_select_build_options() -> dict:
    # DeepSelect's device function relies on these flags (its fp32 index
    # arithmetic needs --ftz=false under --use_fast_math); every module that
    # compiles it shares them.
    root = (KERNEL_PATH / "csrc" / "deep_select").resolve()
    return dict(
        extra_include_paths=[str(root), str(root / "kerutils" / "include")],
        extra_dependencies=["flashinfer"],
        extra_cflags=["-DNDEBUG", "-DKERUTILS_IS_BUILD_ON_CUDA"],
        extra_cuda_cflags=[
            "-DNDEBUG",
            "-DKERUTILS_IS_BUILD_ON_CUDA",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-extended-lambda",
            "--use_fast_math",
            "--ftz=false",
        ],
    )


@cache_once
def _jit_deep_select_topk_module(topk: int) -> Module:
    _check_topk(topk)
    args = make_cpp_args(topk)
    return load_jit(
        "deep_select_topk",
        *args,
        cuda_files=["elementwise/deep_select_topk.cuh"],
        cuda_wrappers=[("topk", f"DeepSelectTopKFp32Kernel<{args}>::run")],
        **_deep_select_build_options(),
    )


@cache_once
def _jit_deep_select_topk_region_pack_module(topk: int) -> Module:
    _check_topk(topk)
    args = make_cpp_args(topk)
    return load_jit(
        "deep_select_topk_region_pack",
        *args,
        cuda_files=["elementwise/deep_select_topk_pack.cuh"],
        cuda_wrappers=[
            ("topk_region_pack", f"DeepSelectTopKRegionPackFp32Kernel<{args}>::run")
        ],
        **_deep_select_build_options(),
    )


def deep_select_topk(
    scores: torch.Tensor,
    ends: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    _jit_deep_select_topk_module(out.shape[1]).topk(scores, ends, out)
    return out


def deep_select_topk_region_pack(
    scores: torch.Tensor,
    ends: torch.Tensor,
    topk_ids: torch.Tensor,
    positions: torch.Tensor,
    row_to_req: torch.Tensor,
    req_to_token: torch.Tensor,
    region_ids: torch.Tensor,
    counts: torch.Tensor,
    packed: torch.Tensor,
    causal_lengths: torch.Tensor,
    *,
    use_pdl: bool = False,
) -> None:
    """``deep_select_topk`` and the Step-5 region pack of its ids in one launch.

    ``topk_ids`` gets exactly what ``deep_select_topk(scores, ends, topk_ids)``
    writes; ``region_ids``, ``counts``, ``packed`` and ``causal_lengths`` get what
    the Step-5 ``_pack_regions`` Triton kernel writes from it. ``use_pdl`` launches
    the kernel as a programmatic dependent: it waits on ``griddepcontrol.wait``
    before its first global read.
    """
    _jit_deep_select_topk_region_pack_module(topk_ids.shape[1]).topk_region_pack(
        scores,
        ends,
        topk_ids,
        positions,
        row_to_req,
        req_to_token,
        region_ids,
        counts,
        packed,
        causal_lengths,
        bool(use_pdl),
    )
