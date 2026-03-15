import sys
import types
import unittest
from unittest import mock

import torch

if "triton" not in sys.modules:
    triton_mod = types.ModuleType("triton")
    triton_lang_mod = types.ModuleType("triton.language")
    triton_mod.jit = lambda fn=None, **kwargs: fn
    triton_mod.cdiv = lambda a, b: (a + b - 1) // b
    triton_mod.language = triton_lang_mod
    triton_lang_mod.constexpr = object()
    sys.modules["triton"] = triton_mod
    sys.modules["triton.language"] = triton_lang_mod

from sglang.srt.layers.communicator import (
    CommunicateContext,
    CommunicateWithAllReduceAndLayerNormFn,
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)


class _FakeLayerNorm:
    def __init__(self):
        self.fusion_called = False

    def __call__(self, hidden_states, residual):
        return hidden_states, residual

    def forward_with_allreduce_fusion(self, hidden_states, residual):
        self.fusion_called = True
        return hidden_states, residual


class TestTpaCommunicator(unittest.TestCase):
    def _context(self):
        return CommunicateContext(
            process_group_sizes={
                ScatterMode.SCATTERED: 1,
                ScatterMode.TP_ATTN_FULL: 1,
                ScatterMode.FULL: 2,
            },
            attn_tp_rank=0,
            attn_tp_size=1,
            attn_dp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            tp_size=2,
            tp_rank=0,
        )

    def _layer_communicator(self):
        layer_modes = LayerScatterModes(
            layer_input_mode=ScatterMode.TP_ATTN_FULL,
            attn_mode=ScatterMode.TP_ATTN_FULL,
            mlp_mode=ScatterMode.FULL,
            middle_residual_mode=ScatterMode.TP_ATTN_FULL,
            layer_output_mode=ScatterMode.TP_ATTN_FULL,
        )
        server_args = types.SimpleNamespace(
            speculative_algorithm=None,
            is_attention_tpa_enabled=lambda: False,
        )
        with mock.patch(
            "sglang.srt.layers.communicator.CommunicateContext.init_new",
            return_value=self._context(),
        ), mock.patch(
            "sglang.srt.layers.communicator.get_global_server_args",
            return_value=server_args,
        ):
            return LayerCommunicator(
                layer_scatter_modes=layer_modes,
                input_layernorm=_FakeLayerNorm(),
                post_attention_layernorm=_FakeLayerNorm(),
                allow_reduce_scatter=True,
            )

    def test_tpa_uses_attention_tp_reduce_instead_of_full_tp(self):
        hidden_states = torch.randn(2, 8)
        residual = torch.randn(2, 8)
        layernorm = _FakeLayerNorm()
        server_args = types.SimpleNamespace(
            is_attention_tpa_enabled=lambda: True,
        )

        with mock.patch(
            "sglang.srt.layers.communicator.get_global_server_args",
            return_value=server_args,
        ), mock.patch(
            "sglang.srt.layers.communicator.apply_aiter_all_reduce_fusion",
            return_value=True,
        ), mock.patch(
            "sglang.srt.layers.communicator.apply_flashinfer_allreduce_fusion",
            return_value=True,
        ), mock.patch(
            "sglang.srt.layers.communicator.attn_tp_all_reduce",
            side_effect=lambda x: x + 1,
        ) as attn_reduce, mock.patch(
            "sglang.srt.layers.communicator.tensor_model_parallel_all_reduce",
            side_effect=AssertionError("full TP reduce should not be used for TPA"),
        ):
            out, out_residual = (
                CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual(
                    hidden_states=hidden_states,
                    residual=residual,
                    forward_batch=None,
                    layernorm=layernorm,
                    context=self._context(),
                    residual_input_mode=ScatterMode.TP_ATTN_FULL,
                )
            )

        self.assertEqual(attn_reduce.call_count, 1)
        self.assertFalse(layernorm.fusion_called)
        self.assertTrue(torch.equal(out, hidden_states + 1))
        self.assertTrue(torch.equal(out_residual, residual))

    def test_non_tpa_keeps_full_tp_reduce_path(self):
        hidden_states = torch.randn(2, 8)
        residual = torch.randn(2, 8)
        layernorm = _FakeLayerNorm()
        server_args = types.SimpleNamespace(
            is_attention_tpa_enabled=lambda: False,
        )

        with mock.patch(
            "sglang.srt.layers.communicator.get_global_server_args",
            return_value=server_args,
        ), mock.patch(
            "sglang.srt.layers.communicator.apply_aiter_all_reduce_fusion",
            return_value=False,
        ), mock.patch(
            "sglang.srt.layers.communicator.apply_flashinfer_allreduce_fusion",
            return_value=False,
        ), mock.patch(
            "sglang.srt.layers.communicator.attn_tp_all_reduce",
            side_effect=AssertionError("attention TP reduce should not be used without TPA"),
        ), mock.patch(
            "sglang.srt.layers.communicator.tensor_model_parallel_all_reduce",
            side_effect=lambda x: x + 2,
        ) as tp_reduce:
            out, out_residual = (
                CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual(
                    hidden_states=hidden_states,
                    residual=residual,
                    forward_batch=None,
                    layernorm=layernorm,
                    context=self._context(),
                    residual_input_mode=ScatterMode.TP_ATTN_FULL,
                )
            )

        self.assertEqual(tp_reduce.call_count, 1)
        self.assertTrue(torch.equal(out, hidden_states + 2))
        self.assertTrue(torch.equal(out_residual, residual))

    def test_tpa_postprocess_disables_reduce_scatter_shortcut(self):
        layer = self._layer_communicator()
        forward_batch = types.SimpleNamespace(
            dp_padding_mode=types.SimpleNamespace(is_max_len=lambda: True)
        )
        server_args = types.SimpleNamespace(
            is_attention_tpa_enabled=lambda: True,
            speculative_algorithm=None,
        )

        with mock.patch(
            "sglang.srt.layers.communicator.get_global_server_args",
            return_value=server_args,
        ), mock.patch(
            "sglang.srt.layers.communicator.get_local_dp_buffer",
            return_value=torch.empty(1, 8),
        ), mock.patch(
            "sglang.srt.layers.communicator.dp_reduce_scatter_tensor",
            side_effect=AssertionError(
                "reduce-scatter shortcut should be disabled for TPA"
            ),
        ), mock.patch(
            "sglang.srt.layers.communicator.dp_scatter"
        ) as dp_scatter:
            hidden_states, residual = layer.postprocess_layer(
                torch.randn(2, 8), torch.randn(1, 8), forward_batch
            )

        self.assertEqual(dp_scatter.call_count, 1)
        self.assertEqual(hidden_states.shape, (1, 8))
        self.assertEqual(residual.shape, (1, 8))

    def test_tpa_disables_mlp_allreduce_fusion(self):
        layer = self._layer_communicator()
        forward_batch = types.SimpleNamespace(input_ids=torch.zeros(4, dtype=torch.long))
        server_args = types.SimpleNamespace(
            is_attention_tpa_enabled=lambda: True,
        )

        with mock.patch(
            "sglang.srt.layers.communicator.get_global_server_args",
            return_value=server_args,
        ), mock.patch(
            "sglang.srt.layers.communicator.apply_flashinfer_allreduce_fusion",
            return_value=True,
        ):
            should_fuse = layer.should_fuse_mlp_allreduce_with_next_layer(forward_batch)

        self.assertFalse(should_fuse)

    def test_tpa_prepare_attn_uses_attention_tp_reduce_for_fused_input(self):
        layer = self._layer_communicator()
        layer._communicate_simple_fn = (
            lambda *, hidden_states, forward_batch, context: hidden_states
        )
        hidden_states = torch.randn(2, 8)
        hidden_states._sglang_needs_allreduce_fusion = True
        residual = torch.randn(2, 8)
        server_args = types.SimpleNamespace(
            is_attention_tpa_enabled=lambda: True,
        )

        with mock.patch(
            "sglang.srt.layers.communicator.get_global_server_args",
            return_value=server_args,
        ), mock.patch(
            "sglang.srt.layers.communicator.attn_tp_all_reduce",
            side_effect=lambda x: x + 3,
        ) as attn_reduce, mock.patch(
            "sglang.srt.layers.communicator.tensor_model_parallel_all_reduce",
            side_effect=AssertionError("full TP reduce should not be used for TPA"),
        ):
            out, out_residual = layer.prepare_attn(
                hidden_states, residual, forward_batch=None
            )

        self.assertEqual(attn_reduce.call_count, 1)
        self.assertTrue(torch.equal(out, hidden_states + 3))
        self.assertTrue(torch.equal(out_residual, residual))


if __name__ == "__main__":
    unittest.main()
