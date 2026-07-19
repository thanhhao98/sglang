"""Single-token decode handoff between the router and the MoE runner.

A tiny producer/consumer stash used only on the M == 1 decode fast path:
the radix router (moe_fused_gate_radix) can emit the moe_align_block_size
outputs inside its own kernel; fused_marlin_moe consumes them instead of
launching align_single_token.

The handoff is attempt-and-verify: the consumer checks tensor identity
(data_ptr) and metadata before use, and every producer entry is dropped on
the next set or on consume, so a path that bypasses the consumer (another
runner, a shape change) silently falls back to the regular kernels. All
set/consume calls happen inside one forward, so CUDA-graph capture sees a
consistent producer/consumer pairing (replays re-run the captured kernels
with the same stashed pointers).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

# ---- Router alignment handoff (radix router -> fused_marlin_moe) -----------

_align_key: Optional[int] = None
_align_value: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = None


def compute_single_token_block_size(topk: int, num_experts: int) -> int:
    """Mirror of fused_marlin_moe's block-size selection for M == 1."""
    for block_size_m in [8, 16, 32, 48, 64]:
        if topk / num_experts / block_size_m < 0.9:
            break
    return block_size_m


def stash_alignment(
    topk_ids: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_post: torch.Tensor,
    block_size: int,
) -> None:
    global _align_key, _align_value
    _align_key = topk_ids.data_ptr()
    _align_value = (sorted_ids, expert_ids, num_post, block_size)


def consume_alignment(
    topk_ids: torch.Tensor, block_size: int
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    global _align_key, _align_value
    if _align_value is None or _align_key != topk_ids.data_ptr():
        return None
    sorted_ids, expert_ids, num_post, stashed_block = _align_value
    _align_key = None
    _align_value = None
    if stashed_block != block_size:
        return None
    return sorted_ids, expert_ids, num_post
