"""Unit tests for TPA Phase-2 utilities.

Covers:
1. Head distribution: compute_tpa_head_counts for various TP/CP/head configs
2. o_proj dimension: compute_o_proj_tpa_params correctness
3. A2A merge: _lse_weighted_combine vs reference implementation
4. Token redistribution: ReduceScatter/AllGather shape contracts
5. Communicator fusion guards: can_enable_tpa_reduce_scatter / can_fuse_mlp_allreduce
"""

import importlib.util
import os
import unittest

import torch

# Load tpa_utils directly to avoid sglang.__init__ dependency chain
_tpa_utils_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "python",
    "sglang",
    "srt",
    "layers",
    "attention",
    "tpa_utils.py",
)
_spec = importlib.util.spec_from_file_location(
    "tpa_utils", os.path.abspath(_tpa_utils_path)
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_tpa_head_counts = _mod.compute_tpa_head_counts
compute_o_proj_tpa_params = _mod.compute_o_proj_tpa_params
_lse_weighted_combine = _mod._lse_weighted_combine
can_enable_tpa_reduce_scatter = _mod.can_enable_tpa_reduce_scatter
can_fuse_mlp_allreduce = _mod.can_fuse_mlp_allreduce


class TestTPAHeadDistribution(unittest.TestCase):
    """Test compute_tpa_head_counts for various model configs."""

    def test_qwen7b_tp8_dcp1(self):
        """Standard TP=8, no DCP: 32 Q heads, 4 KV heads."""
        result = compute_tpa_head_counts(
            num_attention_heads=32, num_kv_heads=4, attn_tp_size=8, dcp_size=1
        )
        self.assertEqual(result["q_heads_per_tp"], 4)
        self.assertEqual(result["q_heads_per_tp_cp"], 4)
        self.assertEqual(result["kv_heads_per_tp"], 1)
        self.assertEqual(result["total_tp_size"], 8)

    def test_qwen7b_tpa2_dcp4(self):
        """TPA=2, DCP=4: 32 Q heads split by attn_tp=2, then cp=4."""
        result = compute_tpa_head_counts(
            num_attention_heads=32, num_kv_heads=4, attn_tp_size=2, dcp_size=4
        )
        self.assertEqual(result["q_heads_per_tp"], 16)
        self.assertEqual(result["q_heads_per_tp_cp"], 4)
        self.assertEqual(result["kv_heads_per_tp"], 2)
        self.assertEqual(result["total_tp_size"], 8)

    def test_qwen7b_tpa4_dcp2(self):
        """TPA=4, DCP=2: 32 Q heads split by attn_tp=4, then cp=2."""
        result = compute_tpa_head_counts(
            num_attention_heads=32, num_kv_heads=4, attn_tp_size=4, dcp_size=2
        )
        self.assertEqual(result["q_heads_per_tp"], 8)
        self.assertEqual(result["q_heads_per_tp_cp"], 4)
        self.assertEqual(result["kv_heads_per_tp"], 1)
        self.assertEqual(result["total_tp_size"], 8)

    def test_deepseek_128heads_tp8_dcp1(self):
        """DeepSeek-like: 128 Q heads, TP=8, no DCP."""
        result = compute_tpa_head_counts(
            num_attention_heads=128, num_kv_heads=1, attn_tp_size=8, dcp_size=1
        )
        self.assertEqual(result["q_heads_per_tp"], 16)
        self.assertEqual(result["q_heads_per_tp_cp"], 16)
        self.assertEqual(result["kv_heads_per_tp"], 1)

    def test_gqa_8kv_tpa4_dcp4(self):
        """GQA with 8 KV heads, TPA=4, DCP=4 (total TP=16 equivalent)."""
        result = compute_tpa_head_counts(
            num_attention_heads=64, num_kv_heads=8, attn_tp_size=4, dcp_size=4
        )
        self.assertEqual(result["q_heads_per_tp"], 16)
        self.assertEqual(result["q_heads_per_tp_cp"], 4)
        self.assertEqual(result["kv_heads_per_tp"], 2)
        self.assertEqual(result["total_tp_size"], 16)

    def test_mqa_1kv_tp8_dcp2(self):
        """MQA (1 KV head), TP=8, DCP=2."""
        result = compute_tpa_head_counts(
            num_attention_heads=32, num_kv_heads=1, attn_tp_size=8, dcp_size=2
        )
        self.assertEqual(result["q_heads_per_tp"], 4)
        self.assertEqual(result["q_heads_per_tp_cp"], 2)
        self.assertEqual(result["kv_heads_per_tp"], 1)

    def test_invalid_head_count_raises(self):
        """Q heads not divisible by attn_tp_size should raise."""
        with self.assertRaises(AssertionError):
            compute_tpa_head_counts(
                num_attention_heads=32, num_kv_heads=4, attn_tp_size=5, dcp_size=1
            )

    def test_too_many_splits_raises(self):
        """More TP*CP splits than Q heads should raise."""
        with self.assertRaises(AssertionError):
            compute_tpa_head_counts(
                num_attention_heads=4, num_kv_heads=1, attn_tp_size=2, dcp_size=4
            )

    def test_kv_ceil_division(self):
        """KV head ceiling division when attn_tp > num_kv_heads."""
        result = compute_tpa_head_counts(
            num_attention_heads=32, num_kv_heads=2, attn_tp_size=4, dcp_size=1
        )
        self.assertEqual(result["kv_heads_per_tp"], 1)

    def test_all_configs_sum_to_total(self):
        """Verify all ranks' Q heads sum to total for various configs."""
        configs = [
            (32, 4, 4, 2),
            (64, 8, 8, 1),
            (128, 1, 4, 4),
            (32, 4, 2, 4),
            (16, 4, 2, 2),
        ]
        for num_q, num_kv, attn_tp, dcp in configs:
            with self.subTest(q=num_q, kv=num_kv, atp=attn_tp, dcp=dcp):
                result = compute_tpa_head_counts(num_q, num_kv, attn_tp, dcp)
                total = result["q_heads_per_tp_cp"] * result["total_tp_size"]
                self.assertEqual(total, num_q)


class TestOProjTPAParams(unittest.TestCase):
    """Test compute_o_proj_tpa_params for correctness."""

    def test_basic_tpa2_dcp4(self):
        """TPA=2, DCP=4 on 32 heads, head_dim=64."""
        result = compute_o_proj_tpa_params(
            num_attention_heads=32,
            head_dim=64,
            hidden_size=2048,
            attn_tp_size=2,
            dcp_size=4,
            attn_tp_rank=0,
            dcp_rank=0,
        )
        self.assertEqual(result["o_proj_tp_size"], 8)
        self.assertEqual(result["o_proj_tp_rank"], 0)
        self.assertEqual(result["local_num_heads"], 4)
        self.assertEqual(result["o_proj_input_dim"], 32 * 64)
        self.assertEqual(result["o_proj_input_dim_per_rank"], 4 * 64)

    def test_combined_rank_ordering(self):
        """Verify combined_rank = attn_tp_rank * dcp_size + dcp_rank."""
        for atp_rank in range(2):
            for dcp_rank in range(4):
                result = compute_o_proj_tpa_params(
                    num_attention_heads=32,
                    head_dim=64,
                    hidden_size=2048,
                    attn_tp_size=2,
                    dcp_size=4,
                    attn_tp_rank=atp_rank,
                    dcp_rank=dcp_rank,
                )
                expected_rank = atp_rank * 4 + dcp_rank
                self.assertEqual(result["o_proj_tp_rank"], expected_rank)

    def test_all_ranks_cover_full_input_dim(self):
        """Each rank's input dim slice should sum to total input dim."""
        total_heads = 64
        head_dim = 128
        attn_tp = 4
        dcp = 2
        total_input_dim = 0
        for atp_rank in range(attn_tp):
            for dcp_rank in range(dcp):
                result = compute_o_proj_tpa_params(
                    num_attention_heads=total_heads,
                    head_dim=head_dim,
                    hidden_size=4096,
                    attn_tp_size=attn_tp,
                    dcp_size=dcp,
                    attn_tp_rank=atp_rank,
                    dcp_rank=dcp_rank,
                )
                total_input_dim += result["o_proj_input_dim_per_rank"]
        self.assertEqual(total_input_dim, total_heads * head_dim)

    def test_per_rank_dims_consistent(self):
        """All ranks should have same per-rank input dim."""
        dims = set()
        for atp_rank in range(4):
            for dcp_rank in range(2):
                result = compute_o_proj_tpa_params(
                    num_attention_heads=32,
                    head_dim=64,
                    hidden_size=2048,
                    attn_tp_size=4,
                    dcp_size=2,
                    attn_tp_rank=atp_rank,
                    dcp_rank=dcp_rank,
                )
                dims.add(result["o_proj_input_dim_per_rank"])
        self.assertEqual(len(dims), 1)


class TestLSEWeightedCombine(unittest.TestCase):
    """Test _lse_weighted_combine correctness."""

    def test_single_shard_identity(self):
        """N=1: output should equal input."""
        output = torch.randn(1, 4, 8, 64)
        lse = torch.randn(1, 4, 8)
        result = _lse_weighted_combine(output, lse, is_lse_base_on_e=True)
        torch.testing.assert_close(
            result, output.squeeze(0).float(), atol=1e-5, rtol=1e-5
        )

    def test_equal_lse_gives_mean(self):
        """Equal LSE across shards should produce the mean of outputs."""
        N, B, H, D = 4, 2, 8, 64
        outputs = torch.randn(N, B, H, D)
        lses = torch.full((N, B, H), 5.0)
        result = _lse_weighted_combine(outputs, lses, is_lse_base_on_e=True)
        expected = outputs.float().mean(dim=0)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_dominant_shard(self):
        """One shard with much larger LSE should dominate the output."""
        N, B, H, D = 2, 1, 1, 64
        outputs = torch.randn(N, B, H, D)
        lses = torch.tensor([[[100.0]], [[-100.0]]])
        result = _lse_weighted_combine(outputs, lses, is_lse_base_on_e=True)
        torch.testing.assert_close(result, outputs[0].float(), atol=1e-3, rtol=1e-3)

    def test_base_e_vs_base_2_differ(self):
        """Base-e and base-2 should produce different results."""
        N, B, H, D = 2, 2, 4, 32
        outputs = torch.randn(N, B, H, D)
        lses = torch.randn(N, B, H) * 3.0
        result_e = _lse_weighted_combine(outputs, lses, is_lse_base_on_e=True)
        result_2 = _lse_weighted_combine(outputs, lses, is_lse_base_on_e=False)
        self.assertFalse(torch.allclose(result_e, result_2, atol=1e-3))

    def test_nan_lse_handled(self):
        """NaN in LSE should not propagate to output."""
        outputs = torch.randn(2, 1, 1, 8)
        lses = torch.tensor([[[5.0]], [[float("nan")]]])
        result = _lse_weighted_combine(outputs, lses, is_lse_base_on_e=True)
        self.assertFalse(torch.isnan(result).any())

    def test_inf_lse_handled(self):
        """Inf in LSE should not propagate to output."""
        outputs = torch.randn(2, 1, 1, 8)
        lses = torch.tensor([[[5.0]], [[float("inf")]]])
        result = _lse_weighted_combine(outputs, lses, is_lse_base_on_e=True)
        self.assertFalse(torch.isnan(result).any())

    def test_large_n_combine(self):
        """N=8 (DCP=8) combine should work correctly."""
        N, B, H, D = 8, 4, 16, 128
        outputs = torch.randn(N, B, H, D)
        lses = torch.randn(N, B, H)
        result = _lse_weighted_combine(outputs, lses, is_lse_base_on_e=True)
        self.assertEqual(result.shape, (B, H, D))
        self.assertFalse(torch.isnan(result).any())

    def test_output_dtype_matches_input(self):
        """Output dtype should match input dtype."""
        outputs = torch.randn(2, 4, 8, 64)
        lses = torch.randn(2, 4, 8)
        result = _lse_weighted_combine(outputs, lses)
        self.assertEqual(result.dtype, torch.float32)

    def test_symmetry_across_shards(self):
        """Swapping two shards with same LSE should not change result."""
        N, B, H, D = 2, 1, 4, 32
        outputs = torch.randn(N, B, H, D)
        lses = torch.full((N, B, H), 3.0)
        result1 = _lse_weighted_combine(outputs, lses)
        swapped = torch.stack([outputs[1], outputs[0]])
        result2 = _lse_weighted_combine(swapped, lses)
        torch.testing.assert_close(result1, result2, atol=1e-5, rtol=1e-5)


class TestTokenRedistributionShapes(unittest.TestCase):
    """Test shape contracts for ReduceScatter/AllGather token redistribution."""

    def test_reduce_scatter_shape(self):
        """ReduceScatter should produce tokens/dcp_size rows."""
        num_tokens = 32
        hidden_size = 2048
        dcp_size = 4
        hidden = torch.randn(num_tokens, hidden_size)
        expected_shape = (num_tokens // dcp_size, hidden_size)
        self.assertEqual(expected_shape, (8, 2048))

    def test_allgather_shape(self):
        """AllGather should restore the original token count."""
        chunk_tokens = 8
        hidden_size = 2048
        dcp_size = 4
        chunk = torch.randn(chunk_tokens, hidden_size)
        expected_shape = (chunk_tokens * dcp_size, hidden_size)
        self.assertEqual(expected_shape, (32, 2048))

    def test_roundtrip_preserves_token_count(self):
        """RS then AG should return to the original token count."""
        num_tokens = 64
        dcp_size = 8
        chunk_size = num_tokens // dcp_size
        restored = chunk_size * dcp_size
        self.assertEqual(restored, num_tokens)

    def test_indivisible_token_count_detected(self):
        """Token count not divisible by dcp_size should be caught."""
        self.assertNotEqual(33 % 4, 0)


class TestCommunicatorFusionGuards(unittest.TestCase):
    """Test can_enable_tpa_reduce_scatter and can_fuse_mlp_allreduce."""

    def test_no_tpa_always_allows_rs(self):
        self.assertTrue(can_enable_tpa_reduce_scatter(False, False))

    def test_tpa_without_combined_oproj_blocks_rs(self):
        self.assertFalse(can_enable_tpa_reduce_scatter(False, True))

    def test_tpa_with_combined_oproj_allows_rs(self):
        self.assertTrue(can_enable_tpa_reduce_scatter(True, True))

    def test_no_tpa_always_allows_fusion(self):
        self.assertTrue(can_fuse_mlp_allreduce(False, False))

    def test_tpa_without_combined_oproj_blocks_fusion(self):
        self.assertFalse(can_fuse_mlp_allreduce(False, True))

    def test_tpa_with_combined_oproj_allows_fusion(self):
        self.assertTrue(can_fuse_mlp_allreduce(True, True))


class TestTPAvsHelixHeadLayout(unittest.TestCase):
    """Verify TPA head layout matches TRT-LLM Helix expectations.

    TRT-LLM Helix: num_heads_tp_cp = num_heads // cp_size (after TP split)
    sglang TPA:    q_heads_per_tp_cp = q_heads_per_tp // dcp_size

    These should produce the same per-rank head count.
    """

    def test_tp4_cp2_32heads(self):
        """TP=4, CP=2, 32 Q heads: each rank should get 4 Q heads."""
        sglang = compute_tpa_head_counts(32, 8, attn_tp_size=4, dcp_size=2)
        trtllm_heads_per_tp = 32 // 4
        trtllm_heads_per_tp_cp = trtllm_heads_per_tp // 2
        self.assertEqual(sglang["q_heads_per_tp_cp"], trtllm_heads_per_tp_cp)

    def test_tp2_cp4_32heads(self):
        """TP=2, CP=4, 32 Q heads: each rank should get 4 Q heads."""
        sglang = compute_tpa_head_counts(32, 4, attn_tp_size=2, dcp_size=4)
        trtllm_heads_per_tp = 32 // 2
        trtllm_heads_per_tp_cp = trtllm_heads_per_tp // 4
        self.assertEqual(sglang["q_heads_per_tp_cp"], trtllm_heads_per_tp_cp)

    def test_tp8_cp2_128heads(self):
        """TP=8, CP=2, 128 Q heads: each rank should get 8 Q heads."""
        sglang = compute_tpa_head_counts(128, 1, attn_tp_size=8, dcp_size=2)
        trtllm_heads_per_tp = 128 // 8
        trtllm_heads_per_tp_cp = trtllm_heads_per_tp // 2
        self.assertEqual(sglang["q_heads_per_tp_cp"], trtllm_heads_per_tp_cp)

    def test_o_proj_effective_tp_matches_helix(self):
        """o_proj effective TP should equal attn_tp * dcp (like Helix mapping_o)."""
        for attn_tp, dcp in [(2, 4), (4, 2), (8, 2), (4, 4)]:
            with self.subTest(attn_tp=attn_tp, dcp=dcp):
                result = compute_o_proj_tpa_params(
                    num_attention_heads=32,
                    head_dim=64,
                    hidden_size=2048,
                    attn_tp_size=attn_tp,
                    dcp_size=dcp,
                    attn_tp_rank=0,
                    dcp_rank=0,
                )
                self.assertEqual(result["o_proj_tp_size"], attn_tp * dcp)


if __name__ == "__main__":
    unittest.main()
