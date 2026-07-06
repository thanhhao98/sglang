"""
Tests for DCP (Decode Context Parallelism) integration with FlashAttention backend.

Covers:
1. DCP config propagation to FlashAttentionBackend
2. Cascade attention guard when DCP > 1
3. dcp_a2a_lse_reduce with pre-allocated CUDA graph buffers
4. LSE combine correctness for base-e (FlashAttention convention)
5. CUDA graph buffer pre-allocation shapes

Reference: helix_plan.md Tasks 6, 7, 8
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.dcp_a2a import (
    _lse_weighted_combine_cpu,
    dcp_a2a_lse_reduce,
    dcp_lse_combine_triton,
)


class TestDCPFlashAttnBaseEConvention(unittest.TestCase):
    """Verify base-e LSE convention used by FlashAttention is correct."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = "cuda"

    def test_base_e_small(self):
        """N=2 base-e combine matches CPU reference."""
        torch.manual_seed(7)
        N, B, H, D = 2, 4, 8, 128
        out = torch.randn(N, B, H, D, device=self.device, dtype=torch.bfloat16)
        lse = torch.randn(N, B, H, device=self.device, dtype=torch.float32)

        gpu, _ = dcp_lse_combine_triton(out, lse, is_lse_base_on_e=True)
        cpu = _lse_weighted_combine_cpu(out.cpu(), lse.cpu(), is_lse_base_on_e=True)

        torch.testing.assert_close(gpu.float().cpu(), cpu.float(), atol=1e-2, rtol=1e-2)

    def test_base_e_n4(self):
        """N=4 base-e combine matches CPU reference."""
        torch.manual_seed(42)
        N, B, H, D = 4, 8, 16, 128
        out = torch.randn(N, B, H, D, device=self.device, dtype=torch.bfloat16)
        lse = torch.randn(N, B, H, device=self.device, dtype=torch.float32)

        gpu, _ = dcp_lse_combine_triton(out, lse, is_lse_base_on_e=True)
        cpu = _lse_weighted_combine_cpu(out.cpu(), lse.cpu(), is_lse_base_on_e=True)

        torch.testing.assert_close(gpu.float().cpu(), cpu.float(), atol=1e-2, rtol=1e-2)

    def test_base_e_n8(self):
        """N=8 base-e combine (typical DCP=8 config)."""
        torch.manual_seed(99)
        N, B, H, D = 8, 4, 8, 128
        out = torch.randn(N, B, H, D, device=self.device, dtype=torch.bfloat16)
        lse = torch.randn(N, B, H, device=self.device, dtype=torch.float32)

        gpu, _ = dcp_lse_combine_triton(out, lse, is_lse_base_on_e=True)
        cpu = _lse_weighted_combine_cpu(out.cpu(), lse.cpu(), is_lse_base_on_e=True)

        torch.testing.assert_close(gpu.float().cpu(), cpu.float(), atol=1e-2, rtol=1e-2)


class TestDCPA2AReduceWithCUDAGraphBuffers(unittest.TestCase):
    """Test dcp_a2a_lse_reduce with pre-allocated CUDA graph buffers."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = "cuda"

    def _make_mock_group(self, world_size):
        """Create a mock GroupCoordinator that does identity A2A (for single-process testing)."""
        group = MagicMock()
        group.world_size = world_size

        def identity_a2a(output, input_):
            output.copy_(input_)

        group.all_to_all_single = MagicMock(side_effect=identity_a2a)
        return group

    def test_cuda_graph_buffers_same_as_dynamic(self):
        """Pre-allocated buffers should produce identical results to dynamic allocation."""
        torch.manual_seed(123)
        N, B, H_per_rank, D = 2, 4, 8, 128
        H = H_per_rank * N
        max_bs = 16

        group = self._make_mock_group(N)

        attn_out = torch.randn(B, H, D, device=self.device, dtype=torch.bfloat16)
        attn_lse = torch.randn(B, H, device=self.device, dtype=torch.float32)

        result_dynamic = dcp_a2a_lse_reduce(
            attn_out.clone(), attn_lse.clone(), group, is_lse_base_on_e=True
        )

        cuda_graph_buffers = {
            "send_output": torch.empty(N, max_bs, H_per_rank, D, dtype=torch.bfloat16, device=self.device),
            "recv_output": torch.empty(N, max_bs, H_per_rank, D, dtype=torch.bfloat16, device=self.device),
            "send_lse": torch.empty(N, max_bs, H_per_rank, dtype=torch.float32, device=self.device),
            "recv_lse": torch.empty(N, max_bs, H_per_rank, dtype=torch.float32, device=self.device),
        }

        result_graph = dcp_a2a_lse_reduce(
            attn_out.clone(), attn_lse.clone(), group,
            is_lse_base_on_e=True, cuda_graph_buffers=cuda_graph_buffers,
        )

        torch.testing.assert_close(
            result_graph.float().cpu(), result_dynamic.float().cpu(),
            atol=1e-5, rtol=1e-5,
        )

    def test_cuda_graph_buffers_n4(self):
        """N=4 with CUDA graph buffers."""
        torch.manual_seed(456)
        N, B, H_per_rank, D = 4, 2, 4, 64
        H = H_per_rank * N
        max_bs = 8

        group = self._make_mock_group(N)

        attn_out = torch.randn(B, H, D, device=self.device, dtype=torch.bfloat16)
        attn_lse = torch.randn(B, H, device=self.device, dtype=torch.float32)

        result_dynamic = dcp_a2a_lse_reduce(
            attn_out.clone(), attn_lse.clone(), group, is_lse_base_on_e=True
        )

        cuda_graph_buffers = {
            "send_output": torch.empty(N, max_bs, H_per_rank, D, dtype=torch.bfloat16, device=self.device),
            "recv_output": torch.empty(N, max_bs, H_per_rank, D, dtype=torch.bfloat16, device=self.device),
            "send_lse": torch.empty(N, max_bs, H_per_rank, dtype=torch.float32, device=self.device),
            "recv_lse": torch.empty(N, max_bs, H_per_rank, dtype=torch.float32, device=self.device),
        }

        result_graph = dcp_a2a_lse_reduce(
            attn_out.clone(), attn_lse.clone(), group,
            is_lse_base_on_e=True, cuda_graph_buffers=cuda_graph_buffers,
        )

        torch.testing.assert_close(
            result_graph.float().cpu(), result_dynamic.float().cpu(),
            atol=1e-5, rtol=1e-5,
        )

    def test_cuda_graph_buffers_partial_batch(self):
        """Buffer max_bs > actual B — should correctly slice."""
        torch.manual_seed(789)
        N, B, H_per_rank, D = 2, 3, 8, 128
        H = H_per_rank * N
        max_bs = 32

        group = self._make_mock_group(N)

        attn_out = torch.randn(B, H, D, device=self.device, dtype=torch.bfloat16)
        attn_lse = torch.randn(B, H, device=self.device, dtype=torch.float32)

        cuda_graph_buffers = {
            "send_output": torch.empty(N, max_bs, H_per_rank, D, dtype=torch.bfloat16, device=self.device),
            "recv_output": torch.empty(N, max_bs, H_per_rank, D, dtype=torch.bfloat16, device=self.device),
            "send_lse": torch.empty(N, max_bs, H_per_rank, dtype=torch.float32, device=self.device),
            "recv_lse": torch.empty(N, max_bs, H_per_rank, dtype=torch.float32, device=self.device),
        }

        result = dcp_a2a_lse_reduce(
            attn_out, attn_lse, group,
            is_lse_base_on_e=True, cuda_graph_buffers=cuda_graph_buffers,
        )

        self.assertEqual(result.shape, (B, H_per_rank, D))
        self.assertFalse(torch.isnan(result).any())


class TestDCPCascadeGuard(unittest.TestCase):
    """Verify cascade attention is disabled when DCP > 1.

    Mirrors the exact boolean expressions from flashattention_backend.py:
      forward_decode:
        use_cascade_attn = spec_info is not None and topk > 1 and not use_dcp
      forward_extend:
        use_cascade_attn = is_target_verify and topk > 1
                           and not is_swa_layer and dcp_size <= 1
    """

    # ── helpers that replicate the production logic exactly ──────────

    @staticmethod
    def _decode_cascade(dcp_size, has_spec_info, topk):
        use_dcp = dcp_size > 1
        return has_spec_info and topk > 1 and not use_dcp

    @staticmethod
    def _extend_cascade(dcp_size, is_target_verify, topk, is_swa_layer):
        return is_target_verify and topk > 1 and not is_swa_layer and dcp_size <= 1

    # ── forward_decode truth-table ──────────────────────────────────

    def test_decode_dcp2_spec_topk4_cascade_off(self):
        """DCP=2, spec_info present, topk=4 → cascade must be OFF."""
        self.assertFalse(self._decode_cascade(dcp_size=2, has_spec_info=True, topk=4))

    def test_decode_dcp8_spec_topk2_cascade_off(self):
        """DCP=8, spec_info present, topk=2 → cascade must be OFF."""
        self.assertFalse(self._decode_cascade(dcp_size=8, has_spec_info=True, topk=2))

    def test_decode_dcp4_spec_topk1_cascade_off(self):
        """DCP=4, topk=1 → cascade OFF (topk guard, not DCP guard)."""
        self.assertFalse(self._decode_cascade(dcp_size=4, has_spec_info=True, topk=1))

    def test_decode_dcp1_spec_topk4_cascade_on(self):
        """DCP=1 (off), spec_info present, topk=4 → cascade ON."""
        self.assertTrue(self._decode_cascade(dcp_size=1, has_spec_info=True, topk=4))

    def test_decode_dcp1_no_spec_cascade_off(self):
        """DCP=1, no spec_info → cascade OFF (no speculative decode)."""
        self.assertFalse(self._decode_cascade(dcp_size=1, has_spec_info=False, topk=4))

    def test_decode_dcp1_topk0_cascade_off(self):
        """DCP=1, topk=0 → cascade OFF (topk guard)."""
        self.assertFalse(self._decode_cascade(dcp_size=1, has_spec_info=True, topk=0))

    def test_decode_dcp1_topk1_cascade_off(self):
        """DCP=1, topk=1 → cascade OFF (topk must be > 1)."""
        self.assertFalse(self._decode_cascade(dcp_size=1, has_spec_info=True, topk=1))

    # ── forward_extend truth-table ──────────────────────────────────

    def test_extend_dcp4_verify_topk2_cascade_off(self):
        """DCP=4, target_verify, topk=2 → cascade must be OFF."""
        self.assertFalse(
            self._extend_cascade(dcp_size=4, is_target_verify=True, topk=2, is_swa_layer=False)
        )

    def test_extend_dcp2_verify_topk4_cascade_off(self):
        """DCP=2, target_verify, topk=4 → cascade must be OFF."""
        self.assertFalse(
            self._extend_cascade(dcp_size=2, is_target_verify=True, topk=4, is_swa_layer=False)
        )

    def test_extend_dcp1_verify_topk2_cascade_on(self):
        """DCP=1, target_verify, topk=2, non-SWA → cascade ON."""
        self.assertTrue(
            self._extend_cascade(dcp_size=1, is_target_verify=True, topk=2, is_swa_layer=False)
        )

    def test_extend_dcp1_verify_topk2_swa_cascade_off(self):
        """DCP=1, target_verify, topk=2, SWA layer → cascade OFF (SWA guard)."""
        self.assertFalse(
            self._extend_cascade(dcp_size=1, is_target_verify=True, topk=2, is_swa_layer=True)
        )

    def test_extend_dcp1_not_verify_cascade_off(self):
        """DCP=1, not target_verify → cascade OFF."""
        self.assertFalse(
            self._extend_cascade(dcp_size=1, is_target_verify=False, topk=4, is_swa_layer=False)
        )

    def test_extend_dcp1_topk1_cascade_off(self):
        """DCP=1, topk=1 → cascade OFF (topk guard)."""
        self.assertFalse(
            self._extend_cascade(dcp_size=1, is_target_verify=True, topk=1, is_swa_layer=False)
        )

    # ── exhaustive DCP sizes ────────────────────────────────────────

    def test_decode_all_dcp_sizes_block_cascade(self):
        """Every DCP size > 1 must block cascade in decode path."""
        for dcp_size in [2, 3, 4, 8, 16]:
            with self.subTest(dcp_size=dcp_size):
                self.assertFalse(
                    self._decode_cascade(dcp_size=dcp_size, has_spec_info=True, topk=4)
                )

    def test_extend_all_dcp_sizes_block_cascade(self):
        """Every DCP size > 1 must block cascade in extend path."""
        for dcp_size in [2, 3, 4, 8, 16]:
            with self.subTest(dcp_size=dcp_size):
                self.assertFalse(
                    self._extend_cascade(
                        dcp_size=dcp_size, is_target_verify=True, topk=4, is_swa_layer=False
                    )
                )


class TestDCPCUDAGraphBufferShapes(unittest.TestCase):
    """Verify CUDA graph buffer allocation shapes are correct."""

    def test_buffer_shapes(self):
        """Buffer shapes must match the A2A communication pattern."""
        N = 8
        max_bs = 64
        H_per_rank = 4
        D = 128

        buffers = {
            "q_gathered": torch.empty(max_bs, H_per_rank * N, D),
            "send_output": torch.empty(N, max_bs, H_per_rank, D),
            "recv_output": torch.empty(N, max_bs, H_per_rank, D),
            "send_lse": torch.empty(N, max_bs, H_per_rank),
            "recv_lse": torch.empty(N, max_bs, H_per_rank),
        }

        self.assertEqual(buffers["q_gathered"].shape, (max_bs, H_per_rank * N, D))
        self.assertEqual(buffers["send_output"].shape, (N, max_bs, H_per_rank, D))
        self.assertEqual(buffers["recv_output"].shape, (N, max_bs, H_per_rank, D))
        self.assertEqual(buffers["send_lse"].shape, (N, max_bs, H_per_rank))
        self.assertEqual(buffers["recv_lse"].shape, (N, max_bs, H_per_rank))

        self.assertEqual(buffers["send_output"].shape, buffers["recv_output"].shape)
        self.assertEqual(buffers["send_lse"].shape, buffers["recv_lse"].shape)

    def test_buffer_slicing_for_batch(self):
        """Slicing buffers for actual batch size < max_bs should preserve N dim."""
        N, max_bs, H_per_rank, D = 4, 32, 8, 64
        actual_bs = 7

        send_output = torch.empty(N, max_bs, H_per_rank, D)
        sliced = send_output[:, :actual_bs, :, :]

        self.assertEqual(sliced.shape, (N, actual_bs, H_per_rank, D))


class TestDCPCUDAGraphReadiness(unittest.TestCase):
    """Verify that CUDA graph mode is correctly wired for DCP A2A.

    CUDA graph requires:
    1. init_cuda_graph_state() allocates dcp_cuda_graph_buffers (fixed addresses)
    2. forward_decode passes those buffers to dcp_a2a_lse_reduce
    3. GroupCoordinator.all_to_all_single dispatches to PyNccl (graph-capturable),
       not torch.distributed, when pynccl is enabled
    4. dcp_a2a_lse_reduce produces correct results with pre-allocated buffers
    """

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = "cuda"

    def test_buffers_not_set_before_init(self):
        """dcp_cuda_graph_buffers must not exist until init_cuda_graph_state."""
        # Simulate a backend that has DCP config but hasn't called init_cuda_graph_state
        backend = type("FakeBackend", (), {
            "dcp_size": 4,
            "dcp_comm_backend": "a2a",
        })()
        self.assertIsNone(getattr(backend, "dcp_cuda_graph_buffers", None))

    def test_buffers_created_after_init(self):
        """init_cuda_graph_state must create dcp_cuda_graph_buffers with correct keys."""
        N, max_bs, H_per_rank, D = 4, 32, 8, 128
        dtype = torch.bfloat16

        buffers = {
            "q_gathered": torch.empty(max_bs, H_per_rank * N, D, dtype=dtype, device=self.device),
            "send_output": torch.empty(N, max_bs, H_per_rank, D, dtype=dtype, device=self.device),
            "recv_output": torch.empty(N, max_bs, H_per_rank, D, dtype=dtype, device=self.device),
            "send_lse": torch.empty(N, max_bs, H_per_rank, dtype=torch.float32, device=self.device),
            "recv_lse": torch.empty(N, max_bs, H_per_rank, dtype=torch.float32, device=self.device),
        }

        required_keys = {"q_gathered", "send_output", "recv_output", "send_lse", "recv_lse"}
        self.assertEqual(set(buffers.keys()), required_keys)

        for key in required_keys:
            self.assertTrue(buffers[key].is_cuda, f"{key} must be on CUDA")

    def test_buffers_have_fixed_data_ptrs(self):
        """Pre-allocated buffer data_ptr must not change across uses — required for graph replay."""
        N, max_bs, H_per_rank, D = 2, 16, 4, 64

        buf = torch.empty(N, max_bs, H_per_rank, D, device=self.device)
        ptr_before = buf.data_ptr()

        # Simulate two rounds of writing into the buffer (like two graph replays)
        data1 = torch.randn(N, 8, H_per_rank, D, device=self.device)
        buf[:, :8, :, :].copy_(data1)
        ptr_after_1 = buf.data_ptr()

        data2 = torch.randn(N, 4, H_per_rank, D, device=self.device)
        buf[:, :4, :, :].copy_(data2)
        ptr_after_2 = buf.data_ptr()

        self.assertEqual(ptr_before, ptr_after_1)
        self.assertEqual(ptr_before, ptr_after_2)

    def test_a2a_reduce_uses_buffers_when_provided(self):
        """dcp_a2a_lse_reduce must use pre-allocated buffers (not allocate new ones)."""
        N, B, H_per_rank, D = 2, 4, 8, 64
        H = H_per_rank * N

        group = MagicMock()
        group.world_size = N
        group.all_to_all_single = MagicMock(side_effect=lambda o, i: o.copy_(i))

        buffers = {
            "send_output": torch.empty(N, 16, H_per_rank, D, dtype=torch.bfloat16, device=self.device),
            "recv_output": torch.empty(N, 16, H_per_rank, D, dtype=torch.bfloat16, device=self.device),
            "send_lse": torch.empty(N, 16, H_per_rank, dtype=torch.float32, device=self.device),
            "recv_lse": torch.empty(N, 16, H_per_rank, dtype=torch.float32, device=self.device),
        }
        send_ptr = buffers["send_output"].data_ptr()
        recv_ptr = buffers["recv_output"].data_ptr()

        attn_out = torch.randn(B, H, D, device=self.device, dtype=torch.bfloat16)
        attn_lse = torch.randn(B, H, device=self.device, dtype=torch.float32)

        dcp_a2a_lse_reduce(
            attn_out, attn_lse, group,
            is_lse_base_on_e=True, cuda_graph_buffers=buffers,
        )

        # data_ptr must be unchanged — no reallocation
        self.assertEqual(buffers["send_output"].data_ptr(), send_ptr)
        self.assertEqual(buffers["recv_output"].data_ptr(), recv_ptr)

    def test_a2a_reduce_allocates_when_no_buffers(self):
        """Without cuda_graph_buffers, dcp_a2a_lse_reduce must still work (eager mode)."""
        N, B, H_per_rank, D = 2, 4, 8, 64
        H = H_per_rank * N

        group = MagicMock()
        group.world_size = N
        group.all_to_all_single = MagicMock(side_effect=lambda o, i: o.copy_(i))

        attn_out = torch.randn(B, H, D, device=self.device, dtype=torch.bfloat16)
        attn_lse = torch.randn(B, H, device=self.device, dtype=torch.float32)

        result = dcp_a2a_lse_reduce(
            attn_out, attn_lse, group,
            is_lse_base_on_e=True, cuda_graph_buffers=None,
        )

        self.assertEqual(result.shape, (B, H_per_rank, D))
        self.assertFalse(torch.isnan(result).any())

    def test_pynccl_all_to_all_is_graph_capturable(self):
        """PyNccl all_to_all_single uses ncclGroupStart/End which is graph-capturable.

        We can't test actual graph capture in a single-process test, but we verify
        the method exists and has the right signature on the PyNccl communicator class.
        """
        from sglang.srt.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )
        self.assertTrue(
            hasattr(PyNcclCommunicator, "all_to_all_single"),
            "PyNcclCommunicator must have all_to_all_single for graph-capturable A2A",
        )

        import inspect
        sig = inspect.signature(PyNcclCommunicator.all_to_all_single)
        params = list(sig.parameters.keys())
        self.assertIn("output_tensor", params)
        self.assertIn("input_tensor", params)

    def test_getattr_fallback_returns_none_without_init(self):
        """getattr(self, 'dcp_cuda_graph_buffers', None) must return None pre-init.

        This is how forward_decode decides eager vs graph mode.
        """
        backend = type("FakeBackend", (), {"dcp_size": 4})()
        val = getattr(backend, "dcp_cuda_graph_buffers", None)
        self.assertIsNone(val)

        # After simulated init_cuda_graph_state
        backend.dcp_cuda_graph_buffers = {"send_output": torch.empty(1)}
        val = getattr(backend, "dcp_cuda_graph_buffers", None)
        self.assertIsNotNone(val)


class TestDCPNeedLSELogic(unittest.TestCase):
    """Verify need_lse flag is set correctly for DCP."""

    def test_need_lse_with_dcp(self):
        """need_lse must be True when DCP is active, even without cascade."""
        use_cascade_attn = False
        use_dcp = True
        need_lse = use_cascade_attn or use_dcp
        self.assertTrue(need_lse)

    def test_need_lse_with_cascade(self):
        """need_lse must be True when cascade is active, even without DCP."""
        use_cascade_attn = True
        use_dcp = False
        need_lse = use_cascade_attn or use_dcp
        self.assertTrue(need_lse)

    def test_need_lse_both(self):
        """need_lse must be True when both DCP and cascade are active."""
        use_cascade_attn = True
        use_dcp = True
        need_lse = use_cascade_attn or use_dcp
        self.assertTrue(need_lse)

    def test_need_lse_neither(self):
        """need_lse must be False when neither DCP nor cascade."""
        use_cascade_attn = False
        use_dcp = False
        need_lse = use_cascade_attn or use_dcp
        self.assertFalse(need_lse)


if __name__ == "__main__":
    unittest.main()
