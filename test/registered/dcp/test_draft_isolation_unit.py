"""CPU tests for DCP speculative-draft isolation and KV id-space sizing."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.dcp import draft_forward_guard
from sglang.srt.mem_cache.kv_cache_configurator import (
    _assert_pool_covers_allocator,
    _draft_pool_size_for_allocator,
)
from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDraftForwardGuard(unittest.TestCase):
    def test_disables_and_restores_attention_dcp(self):
        parallel = get_parallel()
        with parallel.override(
            dcp_enabled=True,
            dcp_size=8,
            dcp_rank=3,
        ):
            self.assertEqual(parallel.attn_dcp_size, 8)
            self.assertEqual(parallel.attn_dcp_rank, 3)
            with draft_forward_guard(True):
                self.assertFalse(parallel.dcp_enabled)
                self.assertEqual(parallel.attn_dcp_size, 1)
                self.assertEqual(parallel.attn_dcp_rank, 0)
            self.assertTrue(parallel.dcp_enabled)
            self.assertEqual(parallel.attn_dcp_size, 8)
            self.assertEqual(parallel.attn_dcp_rank, 3)

    def test_non_draft_is_noop(self):
        parallel = get_parallel()
        with parallel.override(dcp_enabled=True, dcp_size=4, dcp_rank=2):
            with draft_forward_guard(False):
                self.assertTrue(parallel.dcp_enabled)
                self.assertEqual(parallel.attn_dcp_size, 4)


class TestDraftPoolSizing(unittest.TestCase):
    def test_eagle_budget_scales_draft_layers_by_dcp_size(self):
        spec_algorithm = SimpleNamespace(
            is_eagle=lambda: True,
            is_standalone=lambda: False,
            is_dflash_family=lambda: False,
        )
        kvc = SimpleNamespace(
            model_config=SimpleNamespace(),
            layer_info=SimpleNamespace(num_effective_layers=32),
            spec_algorithm=spec_algorithm,
            is_draft_worker=False,
            spec_aux_config=SimpleNamespace(eagle_draft_num_layers=4),
            server_args=SimpleNamespace(dcp_size=8),
        )

        with (
            patch(
                "sglang.srt.model_executor.pool_configurator.mambaish_config",
                return_value=None,
            ),
            patch.object(
                DefaultPoolConfigurator,
                "_compute_cell_size",
                return_value=3200,
            ),
        ):
            configurator = DefaultPoolConfigurator(kvc)

        self.assertEqual(configurator._cell_size, 6400)

    def test_widening_covers_allocator_padding_page(self):
        allocator = SimpleNamespace(size=800, page_size=8)
        widened = _draft_pool_size_for_allocator(
            configured_size=100,
            allocator=allocator,
            pool_page_size=1,
        )
        self.assertEqual(widened, 807)

        _assert_pool_covers_allocator(
            SimpleNamespace(size=widened, page_size=1), allocator
        )
        with self.assertRaisesRegex(AssertionError, "does not cover"):
            _assert_pool_covers_allocator(
                SimpleNamespace(size=widened - 1, page_size=1), allocator
            )


if __name__ == "__main__":
    unittest.main()
