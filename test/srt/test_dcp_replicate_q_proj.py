"""Unit tests for DCP replicate Q projection feature.

Tests:
1. ServerArgs: --dcp-replicate-q-proj field existence, defaults, validation
2. Init logic: dcp_replicate flag, q_tp_rank/q_tp_size computation
3. Forward methods: Q head count, AllGather Q skip, replicated kv_b_proj slicing
4. Weight loading: _slice_replicated_w_vc correctness
"""

import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.server_args import ServerArgs

_mock_device = patch("sglang.srt.server_args.get_device", return_value="cuda")
_mock_device.start()


class TestReplicateQProjFieldDefaults(unittest.TestCase):
    """Verify dcp_replicate_q_proj field exists with correct default."""

    def test_field_exists(self):
        fields = {f.name for f in dataclasses.fields(ServerArgs)}
        self.assertIn("dcp_replicate_q_proj", fields)

    def test_default_is_false(self):
        self.assertFalse(ServerArgs.dcp_replicate_q_proj)


class TestReplicateQProjValidation(unittest.TestCase):
    """Verify --dcp-replicate-q-proj validation logic."""

    def _make_args(self, **kwargs):
        defaults = dict(
            model_path="dummy",
            dcp_size=1,
            dcp_comm_backend="ag_rs",
            dcp_replicate_q_proj=False,
        )
        defaults.update(kwargs)
        return ServerArgs(**defaults)

    def test_replicate_q_proj_requires_dcp_size_gt_1(self):
        args = self._make_args(dcp_size=1, dcp_replicate_q_proj=True)
        with self.assertRaises(ValueError):
            args._handle_context_parallelism()

    def test_replicate_q_proj_with_dcp_size_2_passes(self):
        args = self._make_args(dcp_size=2, dcp_replicate_q_proj=True)
        args._handle_context_parallelism()
        self.assertTrue(args.dcp_replicate_q_proj)

    def test_replicate_q_proj_with_dcp_size_8_passes(self):
        args = self._make_args(dcp_size=8, dcp_replicate_q_proj=True)
        args._handle_context_parallelism()
        self.assertTrue(args.dcp_replicate_q_proj)

    def test_replicate_q_proj_false_with_dcp_size_1_ok(self):
        args = self._make_args(dcp_size=1, dcp_replicate_q_proj=False)
        args._handle_context_parallelism()
        self.assertFalse(args.dcp_replicate_q_proj)

    def test_replicate_q_proj_combinable_with_a2a(self):
        args = self._make_args(
            dcp_size=4, dcp_comm_backend="a2a", dcp_replicate_q_proj=True
        )
        args._handle_context_parallelism()
        self.assertTrue(args.dcp_replicate_q_proj)
        self.assertEqual(args.dcp_comm_backend, "a2a")


class TestReplicateQProjInitLogic(unittest.TestCase):
    """Test DCP replicate flag and tp_rank/tp_size computation logic.

    Mirrors the init logic in DeepseekV2AttentionMLA.__init__:
        dcp_replicate = get_dcp_world_size() > 1 and server_args.dcp_replicate_q_proj
        q_tp_rank = 0 if dcp_replicate else attn_tp_rank
        q_tp_size = 1 if dcp_replicate else attn_tp_size
    """

    @staticmethod
    def _compute_q_tp(dcp_world_size, replicate_q_proj, attn_tp_rank, attn_tp_size):
        dcp_replicate = dcp_world_size > 1 and replicate_q_proj
        q_tp_rank = 0 if dcp_replicate else attn_tp_rank
        q_tp_size = 1 if dcp_replicate else attn_tp_size
        return dcp_replicate, q_tp_rank, q_tp_size

    def test_no_dcp_no_replicate(self):
        flag, rank, size = self._compute_q_tp(
            dcp_world_size=1, replicate_q_proj=False, attn_tp_rank=3, attn_tp_size=8
        )
        self.assertFalse(flag)
        self.assertEqual(rank, 3)
        self.assertEqual(size, 8)

    def test_dcp_without_replicate(self):
        flag, rank, size = self._compute_q_tp(
            dcp_world_size=4, replicate_q_proj=False, attn_tp_rank=2, attn_tp_size=8
        )
        self.assertFalse(flag)
        self.assertEqual(rank, 2)
        self.assertEqual(size, 8)

    def test_dcp_with_replicate(self):
        flag, rank, size = self._compute_q_tp(
            dcp_world_size=8, replicate_q_proj=True, attn_tp_rank=5, attn_tp_size=8
        )
        self.assertTrue(flag)
        self.assertEqual(rank, 0)
        self.assertEqual(size, 1)

    def test_dcp1_with_replicate_no_effect(self):
        flag, rank, size = self._compute_q_tp(
            dcp_world_size=1, replicate_q_proj=True, attn_tp_rank=3, attn_tp_size=8
        )
        self.assertFalse(flag)
        self.assertEqual(rank, 3)
        self.assertEqual(size, 8)


class TestReplicateQProjHeadCount(unittest.TestCase):
    """Test head count logic used in forward methods.

    proj_q_heads = num_heads if dcp_replicate_q_proj else num_local_heads
    """

    @staticmethod
    def _proj_q_heads(num_heads, num_local_heads, dcp_replicate_q_proj):
        return num_heads if dcp_replicate_q_proj else num_local_heads

    def test_replicate_uses_all_heads(self):
        result = self._proj_q_heads(
            num_heads=128, num_local_heads=16, dcp_replicate_q_proj=True
        )
        self.assertEqual(result, 128)

    def test_no_replicate_uses_local_heads(self):
        result = self._proj_q_heads(
            num_heads=128, num_local_heads=16, dcp_replicate_q_proj=False
        )
        self.assertEqual(result, 16)


class TestReplicateQProjSlicing(unittest.TestCase):
    """Test Q/KV head slicing back to local heads after replicated projection."""

    def test_q_slice_back_to_local_heads(self):
        num_heads, num_local_heads = 16, 4
        seq_len = 8
        head_dim = 64
        q_all = torch.randn(seq_len, num_heads, head_dim)
        for tp_rank in range(num_heads // num_local_heads):
            start = tp_rank * num_local_heads
            q_local = q_all[:, start : start + num_local_heads, :].contiguous()
            self.assertEqual(q_local.shape, (seq_len, num_local_heads, head_dim))
            torch.testing.assert_close(
                q_local, q_all[:, start : start + num_local_heads, :]
            )

    def test_kv_slice_back_to_local_heads(self):
        num_heads, num_local_heads = 16, 4
        seq_len = 8
        kv_dim = 192
        kv_all = torch.randn(seq_len, num_heads, kv_dim)
        for tp_rank in range(num_heads // num_local_heads):
            start = tp_rank * num_local_heads
            kv_local = kv_all[:, start : start + num_local_heads, :].contiguous()
            self.assertEqual(kv_local.shape, (seq_len, num_local_heads, kv_dim))

    def test_slice_covers_all_heads(self):
        """Verify each tp_rank's slice is disjoint and covers all heads."""
        num_heads, num_local_heads = 32, 8
        tp_size = num_heads // num_local_heads
        full = torch.arange(num_heads)
        slices = []
        for tp_rank in range(tp_size):
            start = tp_rank * num_local_heads
            slices.append(full[start : start + num_local_heads])
        reconstructed = torch.cat(slices)
        torch.testing.assert_close(reconstructed, full)


class TestAllGatherQSkipLogic(unittest.TestCase):
    """Test the AllGather Q skip decision in forward_absorb_prepare.

    When dcp_replicate_q_proj is True and mode is decode:
      - Skip AllGather, just make contiguous
    When dcp_replicate_q_proj is False:
      - Perform AllGather as normal
    """

    @staticmethod
    def _should_allgather_q(dcp_replicate_q_proj, is_decode, dcp_world_size):
        if dcp_world_size <= 1:
            return False
        if is_decode:
            return not dcp_replicate_q_proj
        return False

    def test_decode_with_replicate_skips_allgather(self):
        self.assertFalse(
            self._should_allgather_q(
                dcp_replicate_q_proj=True, is_decode=True, dcp_world_size=8
            )
        )

    def test_decode_without_replicate_does_allgather(self):
        self.assertTrue(
            self._should_allgather_q(
                dcp_replicate_q_proj=False, is_decode=True, dcp_world_size=8
            )
        )

    def test_no_dcp_no_allgather(self):
        self.assertFalse(
            self._should_allgather_q(
                dcp_replicate_q_proj=False, is_decode=True, dcp_world_size=1
            )
        )

    def test_extend_mode_no_allgather(self):
        self.assertFalse(
            self._should_allgather_q(
                dcp_replicate_q_proj=True, is_decode=False, dcp_world_size=8
            )
        )


class TestSliceReplicatedWVc(unittest.TestCase):
    """Test _slice_replicated_w_vc weight slicing logic."""

    def _make_mock_model(self, num_layers, num_heads, num_local_heads, v_head_dim):
        """Create a mock model structure for testing weight slicing."""
        layers = []
        for _ in range(num_layers):
            attn = SimpleNamespace(
                num_local_heads=num_local_heads,
                w_vc=torch.nn.Parameter(
                    torch.randn(num_heads, 512, v_head_dim), requires_grad=False
                ),
                w_scale_v=torch.nn.Parameter(
                    torch.randn(num_heads, 1, 1), requires_grad=False
                ),
            )
            layer = SimpleNamespace(self_attn=attn)
            layers.append(layer)

        model = SimpleNamespace(layers=layers)
        return SimpleNamespace(model=model)

    def _slice_w_vc(self, mock_model, attn_tp_rank):
        """Replicates _slice_replicated_w_vc logic for testing."""
        for layer in mock_model.model.layers:
            if not hasattr(layer, "self_attn"):
                continue
            self_attn = layer.self_attn
            if not hasattr(self_attn, "w_vc") or self_attn.w_vc is None:
                continue
            start = attn_tp_rank * self_attn.num_local_heads
            end = start + self_attn.num_local_heads
            self_attn.w_vc = torch.nn.Parameter(
                self_attn.w_vc[start:end].contiguous(), requires_grad=False
            )
            if hasattr(self_attn, "w_scale_v") and self_attn.w_scale_v is not None:
                self_attn.w_scale_v = torch.nn.Parameter(
                    self_attn.w_scale_v[start:end].contiguous(), requires_grad=False
                )

    def test_slices_w_vc_to_local_heads(self):
        num_heads, num_local_heads, v_head_dim = 16, 4, 128
        mock = self._make_mock_model(2, num_heads, num_local_heads, v_head_dim)
        original_w_vc_0 = mock.model.layers[0].self_attn.w_vc.data.clone()

        self._slice_w_vc(mock, attn_tp_rank=0)

        for layer in mock.model.layers:
            self.assertEqual(
                layer.self_attn.w_vc.shape, (num_local_heads, 512, v_head_dim)
            )
            self.assertEqual(layer.self_attn.w_scale_v.shape, (num_local_heads, 1, 1))

        torch.testing.assert_close(
            mock.model.layers[0].self_attn.w_vc.data,
            original_w_vc_0[:num_local_heads],
        )

    def test_different_tp_ranks_get_different_slices(self):
        num_heads, num_local_heads, v_head_dim = 16, 4, 128
        tp_size = num_heads // num_local_heads
        sliced_w_vcs = []
        for tp_rank in range(tp_size):
            mock = self._make_mock_model(1, num_heads, num_local_heads, v_head_dim)
            torch.manual_seed(42)
            mock.model.layers[0].self_attn.w_vc = torch.nn.Parameter(
                torch.randn(num_heads, 512, v_head_dim), requires_grad=False
            )
            original = mock.model.layers[0].self_attn.w_vc.data.clone()
            self._slice_w_vc(mock, attn_tp_rank=tp_rank)
            sliced = mock.model.layers[0].self_attn.w_vc.data

            start = tp_rank * num_local_heads
            torch.testing.assert_close(
                sliced, original[start : start + num_local_heads]
            )
            sliced_w_vcs.append(sliced.clone())

        for i in range(len(sliced_w_vcs)):
            for j in range(i + 1, len(sliced_w_vcs)):
                self.assertFalse(torch.allclose(sliced_w_vcs[i], sliced_w_vcs[j]))

    def test_no_w_vc_skipped(self):
        """Layers without w_vc should be skipped gracefully."""
        layer_no_wvc = SimpleNamespace(
            self_attn=SimpleNamespace(num_local_heads=4, w_vc=None, w_scale_v=None)
        )
        layer_no_attn = SimpleNamespace()
        mock = SimpleNamespace(
            model=SimpleNamespace(layers=[layer_no_wvc, layer_no_attn])
        )
        self._slice_w_vc(mock, attn_tp_rank=0)

    def test_w_scale_v_none_handled(self):
        """w_scale_v=None should be handled without error."""
        attn = SimpleNamespace(
            num_local_heads=4,
            w_vc=torch.nn.Parameter(torch.randn(16, 512, 128), requires_grad=False),
            w_scale_v=None,
        )
        layer = SimpleNamespace(self_attn=attn)
        mock = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
        self._slice_w_vc(mock, attn_tp_rank=0)
        self.assertEqual(attn.w_vc.shape, (4, 512, 128))
        self.assertIsNone(attn.w_scale_v)


class TestNonDecodeReplicateQSliceBack(unittest.TestCase):
    """Test that non-decode modes with replicated Q slice back to local heads.

    In forward_absorb_prepare, after DCP extend handling:
        if self.dcp_replicate_q_proj and not forward_batch.forward_mode.is_decode():
            start = get_attention_tp_rank() * self.num_local_heads
            q_nope_out = q_nope_out[:, start:start + self.num_local_heads, :]
            q_pe = q_pe[:, start:start + self.num_local_heads, :]
    """

    def test_extend_mode_slices_replicated_q(self):
        num_heads, num_local_heads = 16, 4
        seq_len = 8
        nope_dim, pe_dim = 512, 64

        q_nope = torch.randn(seq_len, num_heads, nope_dim)
        q_pe = torch.randn(seq_len, num_heads, pe_dim)

        tp_rank = 2
        start = tp_rank * num_local_heads
        q_nope_sliced = q_nope[:, start : start + num_local_heads, :].contiguous()
        q_pe_sliced = q_pe[:, start : start + num_local_heads, :].contiguous()

        self.assertEqual(q_nope_sliced.shape, (seq_len, num_local_heads, nope_dim))
        self.assertEqual(q_pe_sliced.shape, (seq_len, num_local_heads, pe_dim))
        torch.testing.assert_close(
            q_nope_sliced, q_nope[:, start : start + num_local_heads, :]
        )

    def test_decode_mode_keeps_all_heads(self):
        """In decode mode, replicated Q is NOT sliced (all heads needed for DCP attention)."""
        num_heads = 16
        seq_len = 1
        nope_dim = 512

        q_nope = torch.randn(seq_len, num_heads, nope_dim)
        q_nope_contiguous = q_nope.contiguous()
        self.assertEqual(q_nope_contiguous.shape, (seq_len, num_heads, nope_dim))


class TestFlashAttentionAllGatherQSkip(unittest.TestCase):
    """Test the AllGather Q skip in flashattention_backend.py.

    if use_dcp and not get_global_server_args().dcp_replicate_q_proj:
        q_reshaped = get_dcp_group().all_gather(q_reshaped.contiguous(), dim=1)
    """

    @staticmethod
    def _should_allgather_q_flash(use_dcp, dcp_replicate_q_proj):
        return use_dcp and not dcp_replicate_q_proj

    def test_dcp_with_replicate_skips(self):
        self.assertFalse(self._should_allgather_q_flash(True, True))

    def test_dcp_without_replicate_gathers(self):
        self.assertTrue(self._should_allgather_q_flash(True, False))

    def test_no_dcp_no_gather(self):
        self.assertFalse(self._should_allgather_q_flash(False, False))

    def test_no_dcp_replicate_no_gather(self):
        self.assertFalse(self._should_allgather_q_flash(False, True))


if __name__ == "__main__":
    unittest.main()
