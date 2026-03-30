"""TPA (Tensor Parallel Attention) Phase-2 utilities.

Implements the key algorithms for TPA performance parity with TRT-LLM Helix:

1. Head distribution: compute Q/KV head counts per rank under TPA layout
2. o_proj dimension: compute the combined TP+CP row-parallel dimension
3. A2A merge for TPA: "same heads, different KV stripes" decode merge
4. Token redistribution: AllGather/ReduceScatter for attention↔FFN handoff
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator


# ---------------------------------------------------------------------------
# 1. Head distribution helpers
# ---------------------------------------------------------------------------


def compute_tpa_head_counts(
    num_attention_heads: int,
    num_kv_heads: int,
    attn_tp_size: int,
    dcp_size: int,
) -> dict:
    """Compute per-rank head counts under TPA layout.

    In TPA, Q heads are split by attn_tp_size first, then optionally further
    split by dcp_size (phase-2D). KV heads use ceiling division for attn_tp.

    Args:
        num_attention_heads: total Q heads in the model
        num_kv_heads: total KV heads in the model
        attn_tp_size: attention tensor parallel size
        dcp_size: decode context parallel size

    Returns:
        dict with:
            q_heads_per_tp: Q heads per attention-TP rank (before CP split)
            q_heads_per_tp_cp: Q heads per rank after TP+CP split
            kv_heads_per_tp: KV heads per attention-TP rank
            total_tp_size: effective TP size (attn_tp_size * dcp_size)
            q_per_kv_group: number of Q heads per KV head on each rank
    """
    assert num_attention_heads % attn_tp_size == 0, (
        f"num_attention_heads ({num_attention_heads}) must be divisible "
        f"by attn_tp_size ({attn_tp_size})"
    )
    q_heads_per_tp = num_attention_heads // attn_tp_size
    q_heads_per_tp_cp = q_heads_per_tp // dcp_size
    kv_heads_per_tp = (num_kv_heads + attn_tp_size - 1) // attn_tp_size
    total_tp_size = attn_tp_size * dcp_size

    assert q_heads_per_tp_cp > 0, (
        f"q_heads_per_tp_cp must be > 0, got {q_heads_per_tp_cp} "
        f"(num_heads={num_attention_heads}, attn_tp={attn_tp_size}, dcp={dcp_size})"
    )

    q_per_kv_group = q_heads_per_tp_cp // kv_heads_per_tp if kv_heads_per_tp > 0 else 0

    return {
        "q_heads_per_tp": q_heads_per_tp,
        "q_heads_per_tp_cp": q_heads_per_tp_cp,
        "kv_heads_per_tp": kv_heads_per_tp,
        "total_tp_size": total_tp_size,
        "q_per_kv_group": q_per_kv_group,
    }


# ---------------------------------------------------------------------------
# 2. o_proj dimension helpers (Phase 2A)
# ---------------------------------------------------------------------------


def compute_o_proj_tpa_params(
    num_attention_heads: int,
    head_dim: int,
    hidden_size: int,
    attn_tp_size: int,
    dcp_size: int,
    attn_tp_rank: int,
    dcp_rank: int,
) -> dict:
    """Compute o_proj parameters for TPA layout (combined TP+CP row-parallel).

    In TRT-LLM Helix, o_proj uses mapping_o with tp_size = tp_size * cp_size.
    Each rank holds num_heads_tp_cp * head_dim input features of the o_proj
    weight, and row-parallel matmul + ReduceScatter distributes tokens.

    Args:
        num_attention_heads: total Q heads
        head_dim: dimension per head
        hidden_size: model hidden size
        attn_tp_size: attention TP size
        dcp_size: DCP size
        attn_tp_rank: this rank's position in the attention TP group
        dcp_rank: this rank's position in the DCP group

    Returns:
        dict with:
            o_proj_input_dim: total input dim of o_proj (num_heads * head_dim)
            o_proj_input_dim_per_rank: per-rank input dim (local heads * head_dim)
            o_proj_tp_size: effective TP size for o_proj (attn_tp * dcp)
            o_proj_tp_rank: this rank's position in the combined TP+CP group
            local_num_heads: number of attention heads this rank owns
    """
    total_tp_size = attn_tp_size * dcp_size
    local_num_heads = num_attention_heads // total_tp_size
    o_proj_input_dim = num_attention_heads * head_dim
    o_proj_input_dim_per_rank = local_num_heads * head_dim

    combined_rank = attn_tp_rank * dcp_size + dcp_rank

    return {
        "o_proj_input_dim": o_proj_input_dim,
        "o_proj_input_dim_per_rank": o_proj_input_dim_per_rank,
        "o_proj_tp_size": total_tp_size,
        "o_proj_tp_rank": combined_rank,
        "local_num_heads": local_num_heads,
    }


# ---------------------------------------------------------------------------
# 3. Token redistribution (Phase 2A: AllGather/ReduceScatter)
# ---------------------------------------------------------------------------


def tpa_reduce_scatter_tokens(
    hidden_states: torch.Tensor,
    dcp_group: "GroupCoordinator",
) -> torch.Tensor:
    """ReduceScatter hidden states across DCP group after o_proj.

    After o_proj produces partial outputs on each rank, ReduceScatter
    sums partials and distributes token chunks so each DCP rank gets
    a distinct slice for FFN processing.

    Args:
        hidden_states: [num_tokens, hidden_size] partial o_proj output
        dcp_group: DCP GroupCoordinator

    Returns:
        [num_tokens // dcp_size, hidden_size] token chunk for this rank
    """
    if dcp_group.world_size == 1:
        return hidden_states

    num_tokens, hidden_size = hidden_states.shape
    assert num_tokens % dcp_group.world_size == 0, (
        f"num_tokens ({num_tokens}) must be divisible by "
        f"dcp_size ({dcp_group.world_size})"
    )
    chunk_size = num_tokens // dcp_group.world_size
    output = torch.empty(
        chunk_size, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device
    )
    dcp_group.reduce_scatter_tensor(output, hidden_states)
    return output


def tpa_allgather_tokens(
    hidden_states: torch.Tensor,
    dcp_group: "GroupCoordinator",
) -> torch.Tensor:
    """AllGather token chunks across DCP group before next attention layer.

    Before attention, each rank needs the full token set. AllGather
    reassembles the token chunks distributed by ReduceScatter.

    Args:
        hidden_states: [num_tokens // dcp_size, hidden_size] local chunk
        dcp_group: DCP GroupCoordinator

    Returns:
        [num_tokens, hidden_size] full token set
    """
    if dcp_group.world_size == 1:
        return hidden_states

    return dcp_group.all_gather(hidden_states, dim=0)


# ---------------------------------------------------------------------------
# 4. A2A merge for TPA decode (Phase 2B)
# ---------------------------------------------------------------------------


def tpa_a2a_decode_merge(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    dcp_group: "GroupCoordinator",
    is_lse_base_on_e: bool = True,
) -> torch.Tensor:
    """TPA-aware A2A decode merge for "same heads, different KV stripes".

    Unlike standard DCP A2A (which exchanges disjoint head slices), TPA ranks
    all own the SAME heads but computed attention over DIFFERENT KV stripes.
    The merge needs to combine partial outputs using LSE-weighted averaging.

    This function uses AllGather + local combine (like TRT-LLM's helix
    approach) rather than the standard DCP A2A head-exchange pattern.

    The key insight: with same heads on all ranks, we AllGather the partial
    outputs and LSEs, then each rank locally combines all partials for its
    own heads. This avoids the head-permutation step of standard A2A.

    Args:
        partial_output: [B, H_local, D] partial attention output (local KV shard)
        partial_lse: [B, H_local] log-sum-exp values
        dcp_group: DCP GroupCoordinator
        is_lse_base_on_e: True for FA (base-e), False for FlashInfer (base-2)

    Returns:
        [B, H_local, D] merged attention output
    """
    if dcp_group.world_size == 1:
        return partial_output

    N = dcp_group.world_size
    B, H_local, D = partial_output.shape

    gathered_output = dcp_group.all_gather(partial_output.contiguous(), dim=0).view(
        N, B, H_local, D
    )

    gathered_lse = dcp_group.all_gather(partial_lse.contiguous(), dim=0).view(
        N, B, H_local
    )

    combined = _lse_weighted_combine(gathered_output, gathered_lse, is_lse_base_on_e)
    return combined


def _lse_weighted_combine(
    partial_outputs: torch.Tensor,
    partial_lses: torch.Tensor,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor:
    """Combine N partial attention outputs using LSE-weighted averaging.

    This is the CPU-friendly reference path. On GPU with Triton available,
    use dcp_lse_combine_triton from dcp_a2a.py for better performance.

    Args:
        partial_outputs: [N, B, H_local, D]
        partial_lses: [N, B, H_local]
        is_lse_base_on_e: base-e (True) or base-2 (False)

    Returns:
        [B, H_local, D] combined output
    """
    partial_outputs = partial_outputs.float()
    partial_lses = partial_lses.float()

    partial_lses = torch.where(
        torch.isnan(partial_lses) | torch.isinf(partial_lses),
        torch.full_like(partial_lses, float("-inf")),
        partial_lses,
    )

    lse_max, _ = partial_lses.max(dim=0)
    lse_max = torch.where(lse_max == float("-inf"), torch.zeros_like(lse_max), lse_max)

    centered = partial_lses - lse_max.unsqueeze(0)
    if is_lse_base_on_e:
        weights = torch.exp(centered)
    else:
        weights = torch.pow(2.0, centered)

    weight_sum = weights.sum(dim=0, keepdim=True)
    weights = weights / weight_sum

    combined = (partial_outputs * weights.unsqueeze(-1)).sum(dim=0)
    return combined.to(partial_outputs.dtype)


# ---------------------------------------------------------------------------
# 5. Communicator fusion guards (Phase 2C)
# ---------------------------------------------------------------------------


def can_enable_tpa_reduce_scatter(
    o_proj_uses_combined_tp_cp: bool,
    is_tpa_enabled: bool,
) -> bool:
    """Determine if reduce-scatter is safe under TPA.

    Phase-1 TPA disables reduce-scatter because o_proj doesn't span the
    combined TP+CP group. Phase-2A fixes this by making o_proj row-parallel
    over tp*cp, after which reduce-scatter is safe again.

    Args:
        o_proj_uses_combined_tp_cp: True if o_proj was created with
            tp_size = attn_tp_size * dcp_size (Phase-2A)
        is_tpa_enabled: True if TPA is active

    Returns:
        True if reduce-scatter can be safely used
    """
    if not is_tpa_enabled:
        return True
    return o_proj_uses_combined_tp_cp


def can_fuse_mlp_allreduce(
    o_proj_uses_combined_tp_cp: bool,
    is_tpa_enabled: bool,
) -> bool:
    """Determine if MLP allreduce+layernorm fusion is safe under TPA.

    Same gating logic as reduce-scatter: only safe when o_proj spans the
    full combined TP+CP group so the output layout matches what fusion expects.

    Args:
        o_proj_uses_combined_tp_cp: True if Phase-2A o_proj is active
        is_tpa_enabled: True if TPA is active

    Returns:
        True if allreduce+layernorm fusion can be used
    """
    if not is_tpa_enabled:
        return True
    return o_proj_uses_combined_tp_cp
