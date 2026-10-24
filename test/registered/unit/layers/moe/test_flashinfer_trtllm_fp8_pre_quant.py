"""Unit tests for the DeepSeekFp8 activation pair of the FlashInfer TRT-LLM FP8 MoE:
quantized here by default, or a caller's pre-quantized pair used as given."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import deepseek_fp8_moe_input
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROWS, HIDDEN, BLOCK = 4, 512, 128
GROUPS = HIDDEN // BLOCK


def column_major_pair(rows=ROWS, hidden=HIDDEN):
    """A pair laid out as per_token_group_quant_fp8(column_major_scales=True) makes it."""
    q = torch.zeros(rows, hidden, dtype=torch.float8_e4m3fn)
    scales = torch.rand(hidden // BLOCK, rows, dtype=torch.float32).t()
    return q, scales


class TestDeepseekFp8MoeInput(CustomTestCase):
    def setUp(self):
        self.hidden = torch.randn(ROWS, HIDDEN, dtype=torch.bfloat16)

    def test_without_a_pair_the_input_is_quantized_here(self):
        q, scales = column_major_pair()
        with patch.object(
            flashinfer_trtllm, "per_token_group_quant_fp8", return_value=(q, scales)
        ) as quant:
            a_q, a_sf_t = deepseek_fp8_moe_input(self.hidden, None, BLOCK)
        quant.assert_called_once_with(self.hidden, BLOCK, column_major_scales=True)
        self.assertIs(a_q, q)
        self.assertEqual(a_sf_t.data_ptr(), scales.data_ptr())
        self.assertEqual(tuple(a_sf_t.shape), (GROUPS, ROWS))

    def test_a_pre_quantized_pair_is_used_as_given(self):
        q, scales = column_major_pair()
        with patch.object(flashinfer_trtllm, "per_token_group_quant_fp8") as quant:
            a_q, a_sf_t = deepseek_fp8_moe_input(self.hidden, (q, scales), BLOCK)
        quant.assert_not_called()
        self.assertIs(a_q, q)
        self.assertTrue(a_sf_t.is_contiguous())
        self.assertEqual(a_sf_t.data_ptr(), scales.data_ptr())
        self.assertEqual(tuple(a_sf_t.shape), (GROUPS, ROWS))

    def test_a_pair_of_another_layout_is_refused(self):
        q, scales = column_major_pair()
        bad = {
            "bf16 codes": (q.to(torch.bfloat16), scales),
            "padded rows": column_major_pair(rows=ROWS + 4),
            "row-major scales": (q, scales.contiguous()),
            "bf16 scales": (q, scales.to(torch.bfloat16)),
            "other group size": (q, torch.rand(HIDDEN // 64, ROWS).t()),
        }
        for name, pair in bad.items():
            with (
                self.subTest(name),
                self.assertRaisesRegex(
                    ValueError, "pre-quantized DeepSeekFp8 MoE input"
                ),
            ):
                deepseek_fp8_moe_input(self.hidden, pair, BLOCK)

    def test_another_pre_quant_object_quantizes_here(self):
        q, scales = column_major_pair()
        with patch.object(
            flashinfer_trtllm, "per_token_group_quant_fp8", return_value=(q, scales)
        ) as quant:
            deepseek_fp8_moe_input(self.hidden, object(), BLOCK)
            deepseek_fp8_moe_input(self.hidden, (q, scales, "ready"), BLOCK)
        self.assertEqual(quant.call_count, 2)


if __name__ == "__main__":
    unittest.main()
