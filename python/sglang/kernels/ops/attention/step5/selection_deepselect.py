# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.step5 import options as step5_options
from sglang.kernels.ops.attention.step5.selection_triton import launch as pack_regions
from sglang.kernels.ops.elementwise.deep_select_topk import (
    DEEP_SELECT_INPUT_STRIDE_ALIGNMENT,
    deep_select_topk,
    deep_select_topk_region_pack,
)

STEP5_HISTORY_REGIONS = 512


@triton.jit
def _prepare_history_lengths(Positions, Requests, Lengths, ROWS: tl.constexpr):
    row = tl.program_id(0) * 128 + tl.arange(0, 128)
    position = tl.load(Positions + row, row < ROWS, other=0)
    request = tl.load(Requests + row, row < ROWS, other=0)
    tl.store(Lengths + row, tl.where(request > 0, position // 8, 0), row < ROWS)


def can_use_step5_selection_deepselect(device: torch.device) -> bool:
    return (
        # TODO(yhyang201): frozen SGLANG_OPT_STEP5_TOPK_DEEPSELECT; prune dead paths.
        step5_options.TOPK_DEEPSELECT
        and device.type == "cuda"
        and torch.version.hip is None
        and torch.cuda.get_device_capability(device)[0] == 10
    )


def deepselect_score_width(score_width: int) -> int:
    align = DEEP_SELECT_INPUT_STRIDE_ALIGNMENT
    return (score_width + align - 1) // align * align


def step5_select_regions_deepselect(
    scores: torch.Tensor,
    positions: torch.Tensor,
    row_to_req: torch.Tensor,
    packed: torch.Tensor,
    counts: torch.Tensor,
    *,
    req_to_token: torch.Tensor,
    causal_lengths: torch.Tensor,
    history_lengths: torch.Tensor,
    topk_ids: torch.Tensor,
    region_ids: torch.Tensor,
    history_ready: bool = False,
) -> None:
    if not history_ready:
        _prepare_history_lengths[(triton.cdiv(positions.numel(), 128),)](
            positions, row_to_req, history_lengths, positions.numel(), num_warps=4
        )
    deep_select_topk(scores, history_lengths, topk_ids)
    pack_regions(
        topk_ids,
        positions,
        row_to_req,
        region_ids,
        counts,
        selected_k=STEP5_HISTORY_REGIONS,
        req_to_token=req_to_token,
        packed=packed,
        causal_lengths=causal_lengths,
    )


def step5_select_regions_deepselect_fused(
    scores: torch.Tensor,
    positions: torch.Tensor,
    row_to_req: torch.Tensor,
    packed: torch.Tensor,
    counts: torch.Tensor,
    *,
    req_to_token: torch.Tensor,
    causal_lengths: torch.Tensor,
    history_lengths: torch.Tensor,
    topk_ids: torch.Tensor,
    region_ids: torch.Tensor,
    history_ready: bool = False,
) -> None:
    """step5_select_regions_deepselect with the top-512 and the pack in one launch.

    Same arguments and outputs, bit for bit: the kernel runs DeepSelect's device
    function with deep_select_topk's config (topk_ids is written identically) and
    then _pack_regions on the selected ids in the same CTA. Opt-in; the default
    Step-5 path keeps the two launches. Unready history lengths are still prepared
    by their own kernel: DeepSelect reads them through the read-only cache, so the
    fused kernel cannot produce them itself.
    """
    if not history_ready:
        _prepare_history_lengths[(triton.cdiv(positions.numel(), 128),)](
            positions, row_to_req, history_lengths, positions.numel(), num_warps=4
        )
    deep_select_topk_region_pack(
        scores,
        history_lengths,
        topk_ids,
        positions,
        row_to_req,
        req_to_token,
        region_ids,
        counts,
        packed,
        causal_lengths,
    )
