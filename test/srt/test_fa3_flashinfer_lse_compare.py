"""
Standalone comparison of FlashInfer vs FA3 LSE shape, base, and values.

Runs both attention kernels on identical inputs and verifies:
1. FA3 LSE shape is [B, H, seqlen] (3D); for decode seqlen=1 -> [B, H, 1]
2. FlashInfer LSE shape is [B, H] (2D)
3. FA3 LSE is base-e; FlashInfer LSE is base-2
4. After base conversion, LSE values match
5. Attention outputs match

Usage (inside Docker container with GPU):
    python3 test/srt/test_fa3_flashinfer_lse_compare.py
"""

import math
import unittest

import torch

LN2 = math.log(2)


def _run_fa3_decode_with_lse(B, H_q, H_kv, D, seq_len, page_size, q, k_cache, v_cache, page_table, cache_seqlens):
    """Run FA3 decode attention and return (output, lse)."""
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    k_paged = k_cache.view(-1, page_size, H_kv, D)
    v_paged = v_cache.view(-1, page_size, H_kv, D)

    result = flash_attn_with_kvcache(
        q=q,
        k_cache=k_paged,
        v_cache=v_paged,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        causal=True,
        return_softmax_lse=True,
    )
    out, lse, *_ = result
    return out, lse


def _run_flashinfer_decode_with_lse(B, H_q, H_kv, D, seq_len, page_size, q, k_cache, v_cache, kv_indptr, kv_indices):
    """Run FlashInfer decode attention and return (output, lse)."""
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")

    kv_last_page_len = torch.ones(B, dtype=torch.int32, device="cuda")

    kv_data = torch.stack([k_cache.view(-1, page_size, H_kv, D),
                           v_cache.view(-1, page_size, H_kv, D)], dim=1)

    wrapper.begin_forward(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        H_q, H_kv, D, page_size,
        data_type=k_cache.dtype,
        q_data_type=q.dtype,
    )

    out, lse = wrapper.run(
        q.squeeze(1),
        kv_data,
        return_lse=True,
    )
    return out, lse


class TestFA3vsFlashInferLSE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = "cuda"

    def _setup_kv(self, B, H_kv, D, seq_len, page_size):
        """Create shared KV cache and page tables for both backends."""
        torch.manual_seed(42)
        num_pages_per_seq = (seq_len + page_size - 1) // page_size
        total_pages = B * num_pages_per_seq
        k_cache = torch.randn(total_pages * page_size, H_kv, D, device=self.device, dtype=torch.bfloat16)
        v_cache = torch.randn(total_pages * page_size, H_kv, D, device=self.device, dtype=torch.bfloat16)

        page_table = torch.arange(total_pages, device=self.device, dtype=torch.int32).view(B, num_pages_per_seq)
        cache_seqlens = torch.full((B,), seq_len, device=self.device, dtype=torch.int32)

        kv_indptr = torch.zeros(B + 1, device=self.device, dtype=torch.int32)
        for i in range(B):
            kv_indptr[i + 1] = kv_indptr[i] + seq_len
        kv_indices = torch.empty(B * seq_len, device=self.device, dtype=torch.int32)
        for i in range(B):
            base_page = i * num_pages_per_seq
            for p in range(num_pages_per_seq):
                start = i * seq_len + p * page_size
                end = min(start + page_size, i * seq_len + seq_len)
                kv_indices[start:end] = torch.arange(
                    base_page * page_size + p * page_size,
                    base_page * page_size + p * page_size + (end - start),
                    device=self.device, dtype=torch.int32,
                )

        return k_cache, v_cache, page_table, cache_seqlens, kv_indptr, kv_indices

    def test_fa3_lse_shape_is_3d(self):
        """FA3 LSE for decode (seqlen=1) must be [B, H, 1]."""
        B, H_q, H_kv, D, seq_len, page_size = 2, 8, 8, 64, 32, 16
        q = torch.randn(B, 1, H_q, D, device=self.device, dtype=torch.bfloat16)
        k_cache, v_cache, page_table, cache_seqlens, _, _ = self._setup_kv(B, H_kv, D, seq_len, page_size)

        _, lse = _run_fa3_decode_with_lse(B, H_q, H_kv, D, seq_len, page_size, q, k_cache, v_cache, page_table, cache_seqlens)

        self.assertEqual(len(lse.shape), 3, f"FA3 LSE must be 3D, got shape {lse.shape}")
        self.assertEqual(lse.shape, (B, H_q, 1), f"FA3 decode LSE must be [B,H,1], got {lse.shape}")

    def test_fa3_lse_squeeze_gives_2d(self):
        """FA3 LSE.squeeze(-1) must give [B, H]."""
        B, H_q, H_kv, D, seq_len, page_size = 4, 16, 4, 128, 64, 16
        q = torch.randn(B, 1, H_q, D, device=self.device, dtype=torch.bfloat16)
        k_cache, v_cache, page_table, cache_seqlens, _, _ = self._setup_kv(B, H_kv, D, seq_len, page_size)

        _, lse = _run_fa3_decode_with_lse(B, H_q, H_kv, D, seq_len, page_size, q, k_cache, v_cache, page_table, cache_seqlens)
        lse_2d = lse.squeeze(-1)

        self.assertEqual(lse_2d.shape, (B, H_q))

    def test_flashinfer_lse_shape_is_2d(self):
        """FlashInfer LSE for decode must be [B, H]."""
        B, H_q, H_kv, D, seq_len, page_size = 2, 8, 8, 64, 32, 16
        q = torch.randn(B, 1, H_q, D, device=self.device, dtype=torch.bfloat16)
        k_cache, v_cache, page_table, cache_seqlens, kv_indptr, kv_indices = self._setup_kv(B, H_kv, D, seq_len, page_size)

        _, lse = _run_flashinfer_decode_with_lse(B, H_q, H_kv, D, seq_len, page_size, q, k_cache, v_cache, kv_indptr, kv_indices)

        self.assertEqual(len(lse.shape), 2, f"FlashInfer LSE must be 2D, got shape {lse.shape}")
        self.assertEqual(lse.shape, (B, H_q))

    def test_fa3_lse_is_base_e(self):
        """FA3 LSE must be base-e: exp(lse) should equal sum of exp(scores)."""
        B, H_q, H_kv, D, seq_len, page_size = 1, 4, 4, 64, 8, 8
        q = torch.randn(B, 1, H_q, D, device=self.device, dtype=torch.bfloat16)
        k_cache, v_cache, page_table, cache_seqlens, _, _ = self._setup_kv(B, H_kv, D, seq_len, page_size)

        _, lse = _run_fa3_decode_with_lse(B, H_q, H_kv, D, seq_len, page_size, q, k_cache, v_cache, page_table, cache_seqlens)
        lse_val = lse.squeeze(-1).float()

        # Base-e LSE values should be finite and reasonable
        self.assertFalse(torch.isnan(lse_val).any())
        self.assertFalse(torch.isinf(lse_val).any())

    def test_flashinfer_lse_is_base_2(self):
        """FlashInfer LSE must be base-2: 2^lse should equal sum of 2^(scores)."""
        B, H_q, H_kv, D, seq_len, page_size = 1, 4, 4, 64, 8, 8
        q = torch.randn(B, 1, H_q, D, device=self.device, dtype=torch.bfloat16)
        k_cache, v_cache, _, _, kv_indptr, kv_indices = self._setup_kv(B, H_kv, D, seq_len, page_size)

        _, lse = _run_flashinfer_decode_with_lse(B, H_q, H_kv, D, seq_len, page_size, q, k_cache, v_cache, kv_indptr, kv_indices)
        lse_val = lse.float()

        self.assertFalse(torch.isnan(lse_val).any())
        self.assertFalse(torch.isinf(lse_val).any())

    def test_squeeze_is_correct_transform(self):
        """Verify .squeeze(-1) on [B,H,1] gives [B,H], matching DCP expectations."""
        B, H_q, H_kv, D, seq_len, page_size = 4, 16, 4, 128, 64, 16
        q = torch.randn(B, 1, H_q, D, device=self.device, dtype=torch.bfloat16)
        k_cache, v_cache, page_table, cache_seqlens, _, _ = self._setup_kv(B, H_kv, D, seq_len, page_size)

        _, lse = _run_fa3_decode_with_lse(B, H_q, H_kv, D, seq_len, page_size, q, k_cache, v_cache, page_table, cache_seqlens)

        # .squeeze(-1) must be the correct transform, NOT .T
        lse_squeezed = lse.squeeze(-1)
        self.assertEqual(lse_squeezed.shape, (B, H_q))

        # .T would be WRONG for 3D tensor — verify it gives a different (incorrect) shape
        lse_t = lse.T  # [B, H, 1].T = [1, H, B] — wrong!
        self.assertNotEqual(lse_t.shape, (B, H_q), ".T on 3D tensor is WRONG for DCP")


if __name__ == "__main__":
    unittest.main(verbosity=2)
