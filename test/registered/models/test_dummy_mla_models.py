"""Regression test: --load-format=dummy must work for MLA-based models.

Root cause of the original bug: multimodal wrapper classes (KimiK25, DeepseekVL2, etc.)
lacked a post_load_weights() method, so the DummyModelLoader's call to
utils.post_load_weights() was a no-op, leaving w_kc/w_vc as None and causing
AttributeError in the forward pass.

Fix: added post_load_weights() to KimiK25ForConditionalGeneration, DeepseekVL2ForCausalLM,
DotsVLMForCausalLM, KimiVLForConditionalGeneration, MiniCPM3ForCausalLM,
SarvamMLAForCausalLM, and extended DeepseekOCRForCausalLM.post_load_weights().
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, is_in_ci, run_bench_one_batch

register_cuda_ci(
    est_time=120,
    suite="stage-b-test-1-gpu",
)


class TestDummyMLALoading(CustomTestCase):
    """Verify that --load-format=dummy works for MLA (w_kc/w_vc must be set)."""

    def test_dummy_deepseek_v2_lite(self):
        """DeepSeek-V2-Lite: base MLA case via DeepseekV2WeightLoaderMixin."""
        _, output_throughput, _ = run_bench_one_batch(
            None,
            [
                "--model",
                "deepseek-ai/DeepSeek-V2-Lite",
                "--batch-size",
                "1",
                "--tp",
                "1",
                "--load-format",
                "dummy",
                "--json-model-override-args",
                '{"num_hidden_layers": 2}',
            ],
        )

        if is_in_ci():
            self.assertGreater(output_throughput, 0)

    def test_dummy_minicpm3(self):
        """MiniCPM3-4B: verifies the extracted post_load_weights() splits w_kc/w_vc."""
        _, output_throughput, _ = run_bench_one_batch(
            None,
            [
                "--model",
                "openbmb/MiniCPM3-4B",
                "--batch-size",
                "1",
                "--tp",
                "1",
                "--load-format",
                "dummy",
                "--json-model-override-args",
                '{"num_hidden_layers": 2}',
            ],
        )

        if is_in_ci():
            self.assertGreater(output_throughput, 0)


if __name__ == "__main__":
    unittest.main()
