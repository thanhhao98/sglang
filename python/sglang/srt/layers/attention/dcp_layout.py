from __future__ import annotations

import torch


def build_dcp_local_page_table(
    global_page_table: torch.Tensor,
    full_seqlens: torch.Tensor,
    max_seq_len_k: int,
    page_size: int,
    dcp_rank: int,
    dcp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the local decode metadata for one DCP rank.

    There are two page-table layouts in the current FA3 path:

    1. Eager decode keeps one column per token position.
    2. CUDA-graph replay keeps one column per KV page.

    DCP needs the same local KV shard for both layouts, so this helper first
    detects which layout it received, then gathers the local page starts using
    the right column space.
    """
    if dcp_size <= 1:
        return global_page_table, full_seqlens.to(torch.int32)

    local_seqlens = ((full_seqlens - dcp_rank - 1) // dcp_size + 1).clamp(min=0)

    max_local_tokens = int(local_seqlens.max().item()) if local_seqlens.numel() else 0
    max_local_pages = (max_local_tokens + page_size - 1) // page_size
    if max_local_pages == 0:
        empty = global_page_table.new_empty((global_page_table.shape[0], 0))
        return empty, local_seqlens.to(torch.int32)

    local_token_starts = torch.arange(
        max_local_pages, device=global_page_table.device, dtype=torch.int64
    ) * page_size
    global_token_positions = dcp_rank + local_token_starts * dcp_size

    # Eager decode keeps a token-location table with one column per token.
    # CUDA-graph replay keeps one column per page.  We need different gather
    # indices for the two cases.
    is_token_table = global_page_table.shape[1] >= max_seq_len_k
    if is_token_table:
        source_cols = global_token_positions
    else:
        source_cols = global_token_positions // page_size

    source_cols = source_cols.clamp(0, global_page_table.shape[1] - 1).to(torch.long)
    local_page_table = global_page_table.index_select(1, source_cols)

    # Eager decode stores one entry per token position, and those entries are
    # the actual token locations used by the KV cache. Keep them unchanged so
    # decode continues to read the same sparse local slots that were written
    # during KV population. CUDA-graph replay uses one entry per page, so that
    # layout still needs remapping into local DCP page ids.
    if not is_token_table:
        valid_mask = local_page_table >= 0
        local_page_table = torch.where(
            valid_mask,
            torch.div(local_page_table, dcp_size, rounding_mode="floor"),
            local_page_table,
        )

    # Zero out rows for padding slots (local_seqlens == 0) to prevent
    # stale page indices from causing OOB access in downstream kernels.
    padding_mask = (local_seqlens == 0).unsqueeze(1)
    if padding_mask.any():
        local_page_table = local_page_table.masked_fill(padding_mask, 0)

    return local_page_table, local_seqlens.to(torch.int32)
