"""
DCP All-to-All communication backend for Decode Context Parallelism.

Alternative to the AG+RS path (cp_lse_ag_out_rs in utils.py).
After local attention produces partial outputs and LSEs for all heads
over a local KV shard, A2A exchanges head partials across DCP ranks,
then a Triton kernel combines them locally using LSE-weighted merging.

Ported from vLLM's dcp_alltoall.py with adaptations for SGLang's
GroupCoordinator API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator


# ---------------------------------------------------------------------------
# Triton kernel: LSE-weighted combine of N partial attention outputs
# ---------------------------------------------------------------------------


@triton.jit
def _dcp_lse_combine_kernel(
    recv_output_ptr,
    recv_lse_ptr,
    out_ptr,
    out_lse_ptr,
    recv_output_stride_N,
    recv_output_stride_B,
    recv_output_stride_H,
    recv_output_stride_D,
    recv_lse_stride_N,
    recv_lse_stride_B,
    recv_lse_stride_H,
    out_stride_B,
    out_stride_H,
    out_stride_D,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    """Combine N partial attention outputs weighted by their LSE values.

    Grid: (B, H_local).
    Each program handles one (batch, head) position across all N shards.

    Two-pass approach:
    Pass 1: find max LSE and weight sum across shards
    Pass 2: accumulate weighted outputs
    """
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)

    lse_base = batch_idx * recv_lse_stride_B + head_idx * recv_lse_stride_H

    # Pass 1: find max LSE across N shards
    lse_max = tl.load(recv_lse_ptr + lse_base).to(tl.float32)
    lse_max = tl.where(
        (lse_max != lse_max) | (lse_max == float("inf")), -float("inf"), lse_max
    )
    for i in tl.static_range(1, N):
        lse_i = tl.load(recv_lse_ptr + lse_base + i * recv_lse_stride_N).to(tl.float32)
        lse_i = tl.where(
            (lse_i != lse_i) | (lse_i == float("inf")), -float("inf"), lse_i
        )
        lse_max = tl.where(lse_i > lse_max, lse_i, lse_max)

    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)

    # Pass 2: accumulate weighted outputs
    weight_sum = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for i in tl.static_range(N):
        lse_i = tl.load(recv_lse_ptr + lse_base + i * recv_lse_stride_N).to(tl.float32)
        lse_i = tl.where(
            (lse_i != lse_i) | (lse_i == float("inf")), -float("inf"), lse_i
        )
        centered = lse_i - lse_max
        if IS_BASE_E:
            w = tl.exp(centered)
        else:
            w = tl.exp2(centered)
        weight_sum += w

        o_offsets = (
            i * recv_output_stride_N
            + batch_idx * recv_output_stride_B
            + head_idx * recv_output_stride_H
            + d_offsets * recv_output_stride_D
        )
        partial_out = tl.load(recv_output_ptr + o_offsets).to(tl.float32)
        acc += partial_out * w

    acc = acc / weight_sum

    out_offsets = (
        batch_idx * out_stride_B + head_idx * out_stride_H + d_offsets * out_stride_D
    )
    tl.store(out_ptr + out_offsets, acc.to(out_ptr.dtype.element_ty))

    if RETURN_LSE:
        if IS_BASE_E:
            global_lse = tl.log(weight_sum) + lse_max
        else:
            global_lse = tl.log2(weight_sum) + lse_max
        out_lse_offset = batch_idx * recv_lse_stride_B + head_idx * recv_lse_stride_H
        tl.store(out_lse_ptr + out_lse_offset, global_lse)


# ---------------------------------------------------------------------------
# Triton launcher
# ---------------------------------------------------------------------------


def dcp_lse_combine_triton(
    recv_output: torch.Tensor,
    recv_lse: torch.Tensor,
    is_lse_base_on_e: bool = True,
    return_lse: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Launch the Triton LSE-combine kernel.

    Args:
        recv_output: [N, B, H_local, D] partial outputs from each DCP rank.
        recv_lse:    [N, B, H_local]    log-sum-exp from each DCP rank.
        is_lse_base_on_e: True if LSE uses base-e (FlashAttention),
                          False if base-2 (FlashInfer).
        return_lse: If True, also return the combined global LSE.

    Returns:
        (combined_output [B, H_local, D], combined_lse [B, H_local] or None)
    """
    N, B, H_local, D = recv_output.shape
    out = torch.empty(
        (B, H_local, D), device=recv_output.device, dtype=recv_output.dtype
    )
    out_lse = (
        torch.empty((B, H_local), device=recv_lse.device, dtype=recv_lse.dtype)
        if return_lse
        else recv_lse.new_empty(0)
    )

    grid = (B, H_local)
    _dcp_lse_combine_kernel[grid](
        recv_output,
        recv_lse,
        out,
        out_lse,
        recv_output.stride(0),
        recv_output.stride(1),
        recv_output.stride(2),
        recv_output.stride(3),
        recv_lse.stride(0),
        recv_lse.stride(1),
        recv_lse.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        N=N,
        HEAD_DIM=D,
        IS_BASE_E=is_lse_base_on_e,
        RETURN_LSE=return_lse,
    )
    return out, (out_lse if return_lse else None)


# ---------------------------------------------------------------------------
# Triton kernel: fused pack output+LSE into A2A send buffer
# ---------------------------------------------------------------------------


@triton.jit
def _pack_a2a_send_kernel(
    # Input: attention output [B, H, D] and LSE [B, H]
    attn_out_ptr,
    attn_lse_ptr,
    # Output: packed send buffer [N, max_bs, H_per_rank, D + LSE_PACK_DIM]
    send_ptr,
    # Strides for attn_out [B, H, D]
    out_stride_B,
    out_stride_H,
    out_stride_D,
    # Strides for attn_lse [B, H]
    lse_stride_B,
    lse_stride_H,
    # Strides for send buffer [N, max_bs, H_per_rank, D+lpd]
    send_stride_N,
    send_stride_B,
    send_stride_H,
    send_stride_D,
    # Dims
    B: tl.constexpr,
    N: tl.constexpr,
    H_PER_RANK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
):
    """Pack attn output [B,H,D] + LSE [B,H] into send buffer [N,B,H/N,D+lpd].

    Grid: (B, H_PER_RANK) — one program per (batch, local_head) pair.
    Each program writes N entries (one per DCP rank) into the send buffer.

    The permutation [B, N, H/N, D] → [N, B, H/N, D] is fused into indexing:
    - Input head index: rank * H_PER_RANK + local_head
    - Output position: send[rank, batch, local_head, :]
    """
    batch_idx = tl.program_id(0)
    head_local = tl.program_id(1)
    d_offsets = tl.arange(0, HEAD_DIM)

    for rank in tl.static_range(N):
        # Source head in the [B, H, D] input
        src_head = rank * H_PER_RANK + head_local

        # Read output values
        in_offset = (
            batch_idx * out_stride_B
            + src_head * out_stride_H
            + d_offsets * out_stride_D
        )
        out_vals = tl.load(attn_out_ptr + in_offset)

        # Write to send buffer [:D] columns
        send_base = (
            rank * send_stride_N
            + batch_idx * send_stride_B
            + head_local * send_stride_H
        )
        tl.store(send_ptr + send_base + d_offsets * send_stride_D, out_vals)

        # Read LSE (fp32) and reinterpret as output dtype for packing
        lse_offset = batch_idx * lse_stride_B + src_head * lse_stride_H
        lse_val = tl.load(attn_lse_ptr + lse_offset)  # fp32 scalar

        # Store LSE as reinterpreted bits into [D:D+lpd] columns
        # For bf16/fp16: 1 fp32 = 2 bf16 elements (LSE_PACK_DIM=2)
        lse_bits = lse_val.to(tl.uint32, bitcast=True)
        if LSE_PACK_DIM == 2:
            lo = (lse_bits & 0xFFFF).to(tl.uint16)
            hi = ((lse_bits >> 16) & 0xFFFF).to(tl.uint16)
            # Store low and high halves separately (Triton doesn't support vector indexing)
            tl.store(
                send_ptr + send_base + HEAD_DIM * send_stride_D,
                lo.to(send_ptr.dtype.element_ty, bitcast=True),
            )
            tl.store(
                send_ptr + send_base + (HEAD_DIM + 1) * send_stride_D,
                hi.to(send_ptr.dtype.element_ty, bitcast=True),
            )
        else:
            # fp32 output: LSE_PACK_DIM=1, store directly
            tl.store(
                send_ptr + send_base + HEAD_DIM * send_stride_D,
                lse_val.to(send_ptr.dtype.element_ty),
            )


@triton.jit
def _unpack_a2a_recv_kernel(
    # Input: packed recv buffer [N, max_bs, H_per_rank, D + LSE_PACK_DIM]
    recv_ptr,
    # Output: LSE staging buffer [N, max_bs, H_per_rank] fp32
    lse_out_ptr,
    # Strides for recv buffer
    recv_stride_N,
    recv_stride_B,
    recv_stride_H,
    recv_stride_D,
    # Strides for lse_out [N, max_bs, H_per_rank]
    lse_stride_N,
    lse_stride_B,
    lse_stride_H,
    # Dims
    B: tl.constexpr,
    N: tl.constexpr,
    H_PER_RANK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
):
    """Unpack LSE from recv buffer [D:] columns into fp32 staging.

    Grid: (B, H_PER_RANK).
    """
    batch_idx = tl.program_id(0)
    head_local = tl.program_id(1)

    for rank in tl.static_range(N):
        recv_base = (
            rank * recv_stride_N
            + batch_idx * recv_stride_B
            + head_local * recv_stride_H
        )

        if LSE_PACK_DIM == 2:
            # Load low and high halves separately (Triton doesn't support vector indexing)
            lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
            hi_raw = tl.load(recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D)
            lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            lse_bits = lo | (hi << 16)
            lse_val = lse_bits.to(tl.float32, bitcast=True)
        else:
            lse_val = tl.load(
                recv_ptr + recv_base + HEAD_DIM * recv_stride_D
            ).to(tl.float32)

        lse_offset = (
            rank * lse_stride_N + batch_idx * lse_stride_B + head_local * lse_stride_H
        )
        tl.store(lse_out_ptr + lse_offset, lse_val)


def _pack_a2a_send(
    attn_out: torch.Tensor,
    attn_lse: torch.Tensor,
    send_combined: torch.Tensor,
    N: int,
    B: int,
    H_per_rank: int,
    D: int,
    lpd: int,
):
    """Fused pack: attn_out[B,H,D] + attn_lse[B,H] → send_combined[N,B,H/N,D+lpd]."""
    grid = (B, H_per_rank)
    _pack_a2a_send_kernel[grid](
        attn_out,
        attn_lse,
        send_combined,
        attn_out.stride(0),
        attn_out.stride(1),
        attn_out.stride(2),
        attn_lse.stride(0),
        attn_lse.stride(1),
        send_combined.stride(0),
        send_combined.stride(1),
        send_combined.stride(2),
        send_combined.stride(3),
        B=B,
        N=N,
        H_PER_RANK=H_per_rank,
        HEAD_DIM=D,
        LSE_PACK_DIM=lpd,
    )


def _unpack_a2a_recv_lse(
    recv_combined: torch.Tensor,
    recv_lse_stg: torch.Tensor,
    N: int,
    B: int,
    H_per_rank: int,
    D: int,
    lpd: int,
):
    """Unpack LSE from recv buffer [D:] into fp32 staging."""
    grid = (B, H_per_rank)
    _unpack_a2a_recv_kernel[grid](
        recv_combined,
        recv_lse_stg,
        recv_combined.stride(0),
        recv_combined.stride(1),
        recv_combined.stride(2),
        recv_combined.stride(3),
        recv_lse_stg.stride(0),
        recv_lse_stg.stride(1),
        recv_lse_stg.stride(2),
        B=B,
        N=N,
        H_PER_RANK=H_per_rank,
        HEAD_DIM=D,
        LSE_PACK_DIM=lpd,
    )


# ---------------------------------------------------------------------------
# Triton kernel: fused unpack + LSE combine (replaces 2 separate kernels)
# ---------------------------------------------------------------------------


@triton.jit
def _fused_unpack_combine_kernel(
    # Input: packed recv buffer [N, max_bs, H_per_rank, D + LSE_PACK_DIM]
    recv_ptr,
    # Output: combined attention output [B, H_local, D]
    out_ptr,
    # Strides for recv buffer [N, max_bs, H_per_rank, D+lpd]
    recv_stride_N,
    recv_stride_B,
    recv_stride_H,
    recv_stride_D,
    # Strides for output [B, H_local, D]
    out_stride_B,
    out_stride_H,
    out_stride_D,
    # Dims
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_BASE_E: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
):
    """Fused unpack + LSE-weighted combine: reads packed recv buffer directly.

    Replaces _unpack_a2a_recv_kernel + dcp_lse_combine_triton (2 kernels → 1).
    Reads output[:D] and packed LSE[D:] from recv buffer, unpacks LSE bits,
    and does weighted combine — all in a single kernel.

    Grid: (B, H_local).
    Each program handles one (batch, head) across all N shards.
    """
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)

    # Pass 1: unpack LSE from all N shards and find max
    lse_max = tl.full([], -float("inf"), dtype=tl.float32)
    for i in tl.static_range(N):
        recv_base = (
            i * recv_stride_N
            + batch_idx * recv_stride_B
            + head_idx * recv_stride_H
        )
        if LSE_PACK_DIM == 2:
            lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
            hi_raw = tl.load(recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D)
            lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            lse_bits = lo | (hi << 16)
            lse_i = lse_bits.to(tl.float32, bitcast=True)
        else:
            lse_i = tl.load(
                recv_ptr + recv_base + HEAD_DIM * recv_stride_D
            ).to(tl.float32)

        # Handle NaN/inf
        lse_i = tl.where(
            (lse_i != lse_i) | (lse_i == float("inf")), -float("inf"), lse_i
        )
        lse_max = tl.where(lse_i > lse_max, lse_i, lse_max)

    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)

    # Pass 2: accumulate weighted outputs (re-read LSE + read output)
    weight_sum = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for i in tl.static_range(N):
        recv_base = (
            i * recv_stride_N
            + batch_idx * recv_stride_B
            + head_idx * recv_stride_H
        )

        # Unpack LSE again (register pressure is better than storing to SRAM)
        if LSE_PACK_DIM == 2:
            lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
            hi_raw = tl.load(recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D)
            lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            lse_bits = lo | (hi << 16)
            lse_i = lse_bits.to(tl.float32, bitcast=True)
        else:
            lse_i = tl.load(
                recv_ptr + recv_base + HEAD_DIM * recv_stride_D
            ).to(tl.float32)

        lse_i = tl.where(
            (lse_i != lse_i) | (lse_i == float("inf")), -float("inf"), lse_i
        )
        centered = lse_i - lse_max
        if IS_BASE_E:
            w = tl.exp(centered)
        else:
            w = tl.exp2(centered)
        weight_sum += w

        # Read output from [:D] columns
        o_offsets = recv_base + d_offsets * recv_stride_D
        partial_out = tl.load(recv_ptr + o_offsets).to(tl.float32)
        acc += partial_out * w

    acc = acc / weight_sum

    out_offsets = (
        batch_idx * out_stride_B + head_idx * out_stride_H + d_offsets * out_stride_D
    )
    tl.store(out_ptr + out_offsets, acc.to(out_ptr.dtype.element_ty))


def dcp_fused_unpack_combine(
    recv_combined: torch.Tensor,
    B: int,
    H_per_rank: int,
    D: int,
    lpd: int,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor:
    """Fused unpack + LSE combine: reads packed recv buffer, outputs combined [B, H_local, D].

    Replaces: _unpack_a2a_recv_lse() + dcp_lse_combine_triton() (2 kernels → 1).
    """
    N = recv_combined.shape[0]
    out = torch.empty(
        (B, H_per_rank, D), device=recv_combined.device, dtype=recv_combined.dtype
    )
    grid = (B, H_per_rank)
    _fused_unpack_combine_kernel[grid](
        recv_combined,
        out,
        recv_combined.stride(0),
        recv_combined.stride(1),
        recv_combined.stride(2),
        recv_combined.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        N=N,
        HEAD_DIM=D,
        IS_BASE_E=is_lse_base_on_e,
        LSE_PACK_DIM=lpd,
    )
    return out


# ---------------------------------------------------------------------------
# Orchestrator — main entry point
# ---------------------------------------------------------------------------


def _lse_pack_dim(output_dtype: torch.dtype) -> int:
    """Number of output-dtype elements needed to store one fp32 LSE value."""
    return torch.finfo(torch.float32).bits // torch.finfo(output_dtype).bits


def dcp_a2a_lse_reduce(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: "GroupCoordinator",
    is_lse_base_on_e: bool = True,
    cuda_graph_buffers: Optional[dict] = None,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """A2A-based DCP reduce: exchange head partials, then local combine.

    Fuses output + LSE into a single all_to_all call by packing fp32 LSE
    as reinterpreted output-dtype elements along the D dimension:
      combined = [N, B, H_per_rank, D + lse_pack_dim]
    This halves the NCCL calls (1 instead of 2 per layer, 27 fewer per step).

    Uses fused Triton kernels for pack and unpack+combine to minimize
    kernel launches (1 pack kernel + 1 A2A + 1 unpack+combine kernel).
    """
    if cp_group.world_size == 1:
        if return_lse:
            return cp_attn_out, cp_attn_lse
        return cp_attn_out

    N = cp_group.world_size
    B, H, D = cp_attn_out.shape
    assert H % N == 0, f"num_heads ({H}) must be divisible by dcp_size ({N})"
    H_per_rank = H // N
    out_dtype = cp_attn_out.dtype
    lpd = _lse_pack_dim(out_dtype)  # 2 for bf16/fp16

    if cuda_graph_buffers is not None:
        # CUDA graph path with pre-allocated fused buffers.
        send_combined = cuda_graph_buffers["send_combined"]
        recv_combined = cuda_graph_buffers["recv_combined"]

        # Fused pack: [B,H,D] + [B,H] → [N, max_bs, H/N, D+lpd]
        _pack_a2a_send(cp_attn_out, cp_attn_lse, send_combined, N, B, H_per_rank, D, lpd)

        # Single fused all_to_all
        cp_group.all_to_all_single(recv_combined.view(-1), send_combined.view(-1))

        # Fused unpack + LSE combine: reads packed recv buffer directly,
        # unpacks LSE bits inline, does weighted combine — single kernel.
        combined = dcp_fused_unpack_combine(
            recv_combined[:, :B, :, :], B, H_per_rank, D, lpd,
            is_lse_base_on_e=is_lse_base_on_e,
        )
    else:
        # Eager path: allocate fused buffer on the fly
        send_combined = torch.empty(
            N, B, H_per_rank, D + lpd,
            dtype=out_dtype, device=cp_attn_out.device,
        )
        recv_combined = torch.empty_like(send_combined)

        # Fused pack
        _pack_a2a_send(cp_attn_out, cp_attn_lse, send_combined, N, B, H_per_rank, D, lpd)

        cp_group.all_to_all_single(recv_combined.view(-1), send_combined.view(-1))

        # Fused unpack + LSE combine
        combined = dcp_fused_unpack_combine(
            recv_combined, B, H_per_rank, D, lpd,
            is_lse_base_on_e=is_lse_base_on_e,
        )

    if return_lse:
        return combined, cp_attn_lse[:, :H_per_rank]
    return combined


# ---------------------------------------------------------------------------
# CPU reference implementation (for unit testing)
# ---------------------------------------------------------------------------


def _lse_weighted_combine_cpu(
    partial_outputs: torch.Tensor,
    partial_lses: torch.Tensor,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor:
    """CPU reference: combine N partial attention outputs using LSE weights.

    Args:
        partial_outputs: [N, B, H_local, D]
        partial_lses:    [N, B, H_local]
        is_lse_base_on_e: base-e (True) or base-2 (False)

    Returns:
        [B, H_local, D] combined output
    """
    N, B, H_local, D = partial_outputs.shape
    partial_outputs = partial_outputs.float()
    partial_lses = partial_lses.float()

    # Sanitize
    partial_lses = torch.where(
        torch.isnan(partial_lses) | torch.isinf(partial_lses),
        torch.full_like(partial_lses, float("-inf")),
        partial_lses,
    )

    # Max LSE for numerical stability: [B, H_local]
    lse_max, _ = partial_lses.max(dim=0)
    lse_max = torch.where(lse_max == float("-inf"), torch.zeros_like(lse_max), lse_max)

    # Compute weights: [N, B, H_local]
    centered = partial_lses - lse_max.unsqueeze(0)
    if is_lse_base_on_e:
        weights = torch.exp(centered)
    else:
        weights = torch.pow(2.0, centered)

    weight_sum = weights.sum(dim=0, keepdim=True)
    weights = weights / weight_sum

    # Weighted sum: [N, B, H_local, D] * [N, B, H_local, 1] -> sum -> [B, H_local, D]
    combined = (partial_outputs * weights.unsqueeze(-1)).sum(dim=0)
    return combined
