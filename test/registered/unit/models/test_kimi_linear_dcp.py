import unittest
from unittest.mock import patch

import torch

from sglang.srt.models.kimi_linear import (
    KimiLinearForCausalLM,
    _get_kda_local_num_heads,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestKimiLinearDCP(CustomTestCase):
    def test_kda_heads_use_global_tp(self):
        self.assertEqual(_get_kda_local_num_heads(64, 4), 16)
        with self.assertRaisesRegex(ValueError, "divisible"):
            _get_kda_local_num_heads(63, 4)

    @patch("sglang.srt.models.kimi_linear.prepare_decode_context_parallel_metadata")
    def test_planner_hook_delegates_all_arguments(self, planner):
        planner.return_value = object()
        kwargs = dict(
            seq_lens=torch.tensor([8]),
            extend_prefix_lens=torch.tensor([4]),
            extend_prefix_lens_cpu=torch.tensor([4]),
            extend_seq_lens=torch.tensor([4]),
            req_pool_indices=torch.tensor([0]),
            req_to_token=torch.arange(16).view(1, 16),
            seq_lens_sum=8,
            kv_buffer_shape=torch.Size([16, 1, 576]),
            kv_cache_dtype=torch.bfloat16,
            kv_cache_device=torch.device("cpu"),
            create_chunked_prefix_cache_kv_indices_fn=object(),
        )
        model = object.__new__(KimiLinearForCausalLM)
        got = model.prepare_context_parallel_metadata_for_dcp(**kwargs)
        self.assertIs(got, planner.return_value)
        planner.assert_called_once_with(**kwargs)


if __name__ == "__main__":
    unittest.main()
