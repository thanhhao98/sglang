"""Unit tests for DCP page table construction with CUDA graph padding.

Validates that build_dcp_local_page_table handles padding slots correctly
when CUDA graph replay pads the batch with dummy entries.

Reproduces the crash scenario: DCP + CUDA graphs + high concurrency (CC512)
where padding slots have req_pool_indices=0, pointing to stale/garbage data
in req_to_token and page_table.
"""

import pytest
import torch

from sglang.srt.layers.attention.dcp_layout import build_dcp_local_page_table


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return "cuda"
    pytest.skip("CUDA not available")


class TestBuildDcpLocalPageTablePadding:
    """Test build_dcp_local_page_table with padding slots (seq_len=0 or 1)."""

    def test_basic_correctness(self, device):
        """Verify basic DCP page table construction with valid data."""
        B, max_pages = 4, 100
        dcp_size, dcp_rank, page_size = 2, 0, 1

        global_page_table = torch.arange(B * max_pages, device=device).view(B, max_pages)
        full_seqlens = torch.tensor([50, 80, 30, 60], dtype=torch.int32, device=device)

        local_pt, local_seqlens = build_dcp_local_page_table(
            global_page_table=global_page_table,
            full_seqlens=full_seqlens,
            max_seq_len_k=80,
            page_size=page_size,
            dcp_rank=dcp_rank,
            dcp_size=dcp_size,
        )

        # Each rank gets ceil((seqlen - rank) / dcp_size) tokens
        expected_local = torch.tensor([25, 40, 15, 30], dtype=torch.int32, device=device)
        assert torch.equal(local_seqlens, expected_local)
        assert local_pt.shape == (B, int(expected_local.max().item()))

    def test_padding_slots_seqlen_zero(self, device):
        """Padding slots with seq_len=0 should produce local_seqlens=0 and not
        cause any out-of-bounds access, even if their page_table rows contain garbage."""
        B = 8  # 4 real + 4 padding
        raw_bs = 4
        max_pages = 200
        dcp_size, dcp_rank, page_size = 2, 0, 1

        # Real rows have valid page indices
        global_page_table = torch.zeros(B, max_pages, dtype=torch.int32, device=device)
        global_page_table[:raw_bs] = torch.arange(max_pages, device=device).unsqueeze(0).expand(raw_bs, -1)

        # Padding rows have GARBAGE values (simulating stale req_to_token data)
        global_page_table[raw_bs:] = torch.randint(
            0, 999999, (B - raw_bs, max_pages), dtype=torch.int32, device=device
        )

        # Real requests have valid seq_lens, padding has 0
        full_seqlens = torch.zeros(B, dtype=torch.int32, device=device)
        full_seqlens[:raw_bs] = torch.tensor([100, 150, 80, 120], dtype=torch.int32)

        local_pt, local_seqlens = build_dcp_local_page_table(
            global_page_table=global_page_table,
            full_seqlens=full_seqlens,
            max_seq_len_k=150,
            page_size=page_size,
            dcp_rank=dcp_rank,
            dcp_size=dcp_size,
        )

        # Padding slots should have local_seqlens = 0
        assert torch.all(local_seqlens[raw_bs:] == 0)
        # Real slots should have correct local_seqlens
        assert local_seqlens[0].item() == 50  # ceil((100-0)/2)
        assert local_seqlens[1].item() == 75  # ceil((150-0)/2)

    def test_padding_slots_seqlen_one(self, device):
        """Padding slots with seq_len=1 (current fill_value) cause DCP to compute
        local_seqlens=1, which then triggers gathering from garbage page_table rows.
        This test verifies the problem exists with fill_value=1."""
        B = 8
        raw_bs = 4
        max_pages = 200
        dcp_size, dcp_rank, page_size = 2, 0, 1

        global_page_table = torch.zeros(B, max_pages, dtype=torch.int32, device=device)
        global_page_table[:raw_bs] = torch.arange(max_pages, device=device).unsqueeze(0).expand(raw_bs, -1)

        # Padding rows: garbage (simulating stale data)
        global_page_table[raw_bs:] = torch.randint(
            0, 999999, (B - raw_bs, max_pages), dtype=torch.int32, device=device
        )

        # fill_value=1 for padding (CURRENT BEHAVIOR — problematic)
        full_seqlens = torch.ones(B, dtype=torch.int32, device=device)
        full_seqlens[:raw_bs] = torch.tensor([100, 150, 80, 120], dtype=torch.int32)

        local_pt, local_seqlens = build_dcp_local_page_table(
            global_page_table=global_page_table,
            full_seqlens=full_seqlens,
            max_seq_len_k=150,
            page_size=page_size,
            dcp_rank=dcp_rank,
            dcp_size=dcp_size,
        )

        # With fill_value=1: padding slots for rank 0 get local_seqlens=1
        # This means DCP will try to gather 1 column from their garbage page_table
        assert local_seqlens[raw_bs].item() == 1  # rank 0: ceil((1-0)/2)=1
        # The gathered value comes from garbage page_table → potential OOB

    def test_padding_slots_seqlen_zero_skips_gather(self, device):
        """With fill_value=0, padding slots get local_seqlens=0, and
        max_local_pages is computed only from real requests → safe."""
        B = 8
        raw_bs = 4
        max_pages = 200
        dcp_size, dcp_rank, page_size = 2, 0, 1

        global_page_table = torch.zeros(B, max_pages, dtype=torch.int32, device=device)
        global_page_table[:raw_bs] = torch.arange(max_pages, device=device).unsqueeze(0).expand(raw_bs, -1)

        # Padding rows: garbage
        global_page_table[raw_bs:] = torch.randint(
            0, 999999, (B - raw_bs, max_pages), dtype=torch.int32, device=device
        )

        # fill_value=0 for padding (PROPOSED FIX)
        full_seqlens = torch.zeros(B, dtype=torch.int32, device=device)
        full_seqlens[:raw_bs] = torch.tensor([100, 150, 80, 120], dtype=torch.int32)

        local_pt, local_seqlens = build_dcp_local_page_table(
            global_page_table=global_page_table,
            full_seqlens=full_seqlens,
            max_seq_len_k=150,
            page_size=page_size,
            dcp_rank=dcp_rank,
            dcp_size=dcp_size,
        )

        # Padding slots have local_seqlens=0 → safe
        assert torch.all(local_seqlens[raw_bs:] == 0)
        # max_local_pages only reflects real requests
        assert local_pt.shape[1] == 75  # max(50, 75, 40, 60)

    def test_all_padding_seqlen_zero(self, device):
        """Edge case: all slots are padding (seq_len=0). Should return empty."""
        B = 4
        max_pages = 100
        dcp_size, dcp_rank, page_size = 2, 0, 1

        global_page_table = torch.zeros(B, max_pages, dtype=torch.int32, device=device)
        full_seqlens = torch.zeros(B, dtype=torch.int32, device=device)

        local_pt, local_seqlens = build_dcp_local_page_table(
            global_page_table=global_page_table,
            full_seqlens=full_seqlens,
            max_seq_len_k=0,
            page_size=page_size,
            dcp_rank=dcp_rank,
            dcp_size=dcp_size,
        )

        assert torch.all(local_seqlens == 0)
        assert local_pt.shape[1] == 0  # No pages needed

    def test_large_batch_with_padding(self, device):
        """Simulate CC512 scenario: bs=512, raw_bs=489."""
        B = 512
        raw_bs = 489
        max_pages = 6000  # page_size=1, max seq len ~6000
        dcp_size, dcp_rank, page_size = 2, 0, 1

        global_page_table = torch.zeros(B, max_pages, dtype=torch.int32, device=device)
        # Real rows: valid sequential page indices
        for i in range(raw_bs):
            seq_len = torch.randint(3000, 5000, (1,)).item()
            global_page_table[i, :seq_len] = torch.arange(seq_len, device=device)

        # Padding rows: garbage from stale req_to_token slot
        global_page_table[raw_bs:] = torch.randint(
            0, 999999, (B - raw_bs, max_pages), dtype=torch.int32, device=device
        )

        # With fill_value=0 (proposed fix)
        full_seqlens = torch.zeros(B, dtype=torch.int32, device=device)
        full_seqlens[:raw_bs] = torch.randint(3000, 5000, (raw_bs,), dtype=torch.int32, device=device)

        # Should not crash
        local_pt, local_seqlens = build_dcp_local_page_table(
            global_page_table=global_page_table,
            full_seqlens=full_seqlens,
            max_seq_len_k=int(full_seqlens[:raw_bs].max().item()),
            page_size=page_size,
            dcp_rank=dcp_rank,
            dcp_size=dcp_size,
        )

        assert torch.all(local_seqlens[raw_bs:] == 0)
        assert local_pt.shape[0] == B
        # max_local_pages should be based on real requests only
        max_real_local = int(local_seqlens[:raw_bs].max().item())
        assert local_pt.shape[1] == (max_real_local + page_size - 1) // page_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
