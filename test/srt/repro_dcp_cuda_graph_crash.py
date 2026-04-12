"""Minimal reproduction of DCP CUDA graph crash at high batch size.

Run under compute-sanitizer to identify the exact OOB kernel:
    compute-sanitizer --tool memcheck --print-limit 5 python3 test/srt/repro_dcp_cuda_graph_crash.py

The crash scenario:
1. CUDA graph captured with bs=512, page_table pointing to pre-allocated buffer
2. During replay, padding slots (e.g. 489..511) have:
   - cache_seqlens = 1 (fill_value)
   - page_table filled with stale indices from req_to_token[0, :]
3. FA3 kernel in graph replay uses these stale page_table values
   to access KV cache → illegal memory access
"""

import torch
import math

try:
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
except ImportError:
    print("sgl_kernel not available, trying flash_attn")
    from flash_attn import flash_attn_with_kvcache


def repro_dcp_cuda_graph_crash():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Model config (CodeQwen1.5-7B-Chat with tp=8)
    bs = 512           # CUDA graph batch size
    raw_bs = 489       # Actual running requests
    num_heads = 4      # per-tp-rank heads
    num_kv_heads = 1   # per-tp-rank KV heads (kv_heads=4, tp=8 → max(1, 4/8)=1)
    head_dim = 128
    page_size = 1
    dcp_size = 2
    dcp_rank = 0
    max_seq_len = 6000  # typical max sequence length at CC512

    # KV cache pool
    num_kv_pages = 2_000_000  # large KV pool
    k_cache = torch.zeros(num_kv_pages, page_size, num_kv_heads, head_dim,
                          dtype=torch.float16, device=device)
    v_cache = torch.zeros(num_kv_pages, page_size, num_kv_heads, head_dim,
                          dtype=torch.float16, device=device)

    # Pre-allocated CUDA graph metadata buffers (stable addresses)
    max_pages = max_seq_len  # page_size=1
    max_local_pages = (max_pages + dcp_size - 1) // dcp_size

    cache_seqlens_buf = torch.zeros(bs, dtype=torch.int32, device=device)
    cu_seqlens_q_buf = torch.arange(bs + 1, dtype=torch.int32, device=device)
    page_table_buf = torch.zeros(bs, max_pages, dtype=torch.int32, device=device)
    dcp_page_table_buf = torch.zeros(bs, max_local_pages, dtype=torch.int32, device=device)
    dcp_cache_seqlens_buf = torch.zeros(bs, dtype=torch.int32, device=device)

    q = torch.randn(bs, num_heads, head_dim, dtype=torch.float16, device=device)

    # ---- CAPTURE phase ----
    # Fill with valid data for capture
    seq_lens_capture = torch.randint(1000, 5000, (bs,), dtype=torch.int32, device=device)
    cache_seqlens_buf.copy_(seq_lens_capture)

    # Fill page_table with valid page indices for ALL slots
    for i in range(bs):
        sl = seq_lens_capture[i].item()
        page_table_buf[i, :sl] = torch.randint(0, num_kv_pages, (sl,),
                                                dtype=torch.int32, device=device)

    # DCP local page table: every other token belongs to this rank
    local_seqlens = ((seq_lens_capture - dcp_rank - 1) // dcp_size + 1).clamp(min=0)
    max_local_tok = int(local_seqlens.max().item())
    max_local_pg = (max_local_tok + page_size - 1) // page_size

    # Build DCP page table from global page table
    local_token_starts = torch.arange(max_local_pg, device=device, dtype=torch.int64) * page_size
    global_positions = dcp_rank + local_token_starts * dcp_size
    source_cols = global_positions.clamp(0, page_table_buf.shape[1] - 1).to(torch.long)
    local_pt = page_table_buf.index_select(1, source_cols)

    dcp_cache_seqlens_buf[:bs].copy_(local_seqlens.to(torch.int32))
    dcp_page_table_buf[:bs, :max_local_pg].copy_(local_pt)

    # Create views that will be captured by the graph.
    # CRITICAL: Use full column width, NOT variable :max_local_pg slice.
    # The CUDA graph records tensor shape at capture time. If replay uses
    # a different max_local_pg, the shape mismatch causes OOB reads.
    dcp_pt_view = dcp_page_table_buf[:bs]  # Full pre-allocated width
    dcp_sl_view = dcp_cache_seqlens_buf[:bs]

    print(f"Capture: bs={bs}, max_local_pg={max_local_pg}, "
          f"dcp_pt_view.shape={dcp_pt_view.shape}, dcp_sl_view.shape={dcp_sl_view.shape}")

    # Warmup run (required before capture)
    torch.cuda.synchronize()
    _ = flash_attn_with_kvcache(
        q=q, k_cache=k_cache, v_cache=v_cache,
        page_table=dcp_pt_view,
        cache_seqlens=dcp_sl_view,
        cu_seqlens_q=cu_seqlens_q_buf,
        max_seqlen_q=1,
        softmax_scale=head_dim ** -0.5,
        causal=False,
        window_size=(-1, -1),
        return_softmax_lse=True,
        num_splits=1,
    )
    torch.cuda.synchronize()

    # Capture graph
    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(graph, stream=s):
            out = flash_attn_with_kvcache(
                q=q, k_cache=k_cache, v_cache=v_cache,
                page_table=dcp_pt_view,
                cache_seqlens=dcp_sl_view,
                cu_seqlens_q=cu_seqlens_q_buf,
                max_seqlen_q=1,
                softmax_scale=head_dim ** -0.5,
                causal=False,
                window_size=(-1, -1),
                return_softmax_lse=True,
                num_splits=1,
            )
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    print("Graph captured successfully")

    # ---- REPLAY phase ----
    # Simulate several decode iterations to trigger the crash
    for iteration in range(50):
        # Simulate varying real batch sizes (sometimes < bs)
        if iteration < 10:
            real_bs = bs  # First 10 iterations: full batch
        else:
            real_bs = raw_bs  # After: padded batch

        # Generate new seq_lens (real requests growing each step)
        real_seq_lens = torch.randint(1000 + iteration * 100,
                                      5000 + iteration * 100,
                                      (real_bs,), dtype=torch.int32, device=device)

        # Padding slots get fill_value=1
        seq_lens_new = torch.ones(bs, dtype=torch.int32, device=device)
        seq_lens_new[:real_bs] = real_seq_lens

        # Update cache_seqlens (in-place to pre-allocated buffer)
        cache_seqlens_buf.copy_(seq_lens_new)

        # Simulate normal_decode_set_metadata: fill page_table
        # Real requests: valid page indices
        for i in range(real_bs):
            sl = min(real_seq_lens[i].item(), max_pages)
            page_table_buf[i, :sl] = torch.randint(0, num_kv_pages, (sl,),
                                                    dtype=torch.int32, device=device)

        # PADDING SLOTS: simulate stale req_to_token[0, :] data
        # Slot 0 was freed → contains GARBAGE (huge indices beyond KV pool)
        if real_bs < bs:
            garbage_pages = torch.randint(num_kv_pages, num_kv_pages * 10,
                                          (bs - real_bs, max_pages),
                                          dtype=torch.int32, device=device)
            page_table_buf[real_bs:] = garbage_pages

        # DCP metadata update (mimics _init_dcp_decode_metadata)
        local_seqlens_new = ((seq_lens_new - dcp_rank - 1) // dcp_size + 1).clamp(min=0)
        max_local_tok_new = int(local_seqlens_new.max().item())
        max_local_pg_new = (max_local_tok_new + page_size - 1) // page_size

        if max_local_pg_new > 0:
            local_starts = torch.arange(max_local_pg_new, device=device, dtype=torch.int64) * page_size
            global_pos = dcp_rank + local_starts * dcp_size
            src_cols = global_pos.clamp(0, page_table_buf.shape[1] - 1).to(torch.long)
            local_pt_new = page_table_buf.index_select(1, src_cols)

            # Update DCP buffers IN-PLACE
            dcp_cache_seqlens_buf[:bs].copy_(local_seqlens_new.to(torch.int32))
            dcp_page_table_buf[:bs, :max_local_pg_new].copy_(local_pt_new)

        # GRAPH REPLAY — this is where the crash should happen
        # The graph was captured with dcp_pt_view shape [512, max_local_pg]
        # But max_local_pg_new may differ → shape mismatch if view changed
        # More importantly: padding rows have garbage page indices
        graph.replay()
        torch.cuda.synchronize()  # Force sync to catch async errors

        if iteration % 10 == 0:
            print(f"  Iteration {iteration}: real_bs={real_bs}, "
                  f"max_local_pg_new={max_local_pg_new}, OK")

    print("All iterations completed without crash!")


if __name__ == "__main__":
    repro_dcp_cuda_graph_crash()
