"""CPU checks of the megakernel binding: the view is the running forward's attention
backend and KV pool, and a source replaced after capture is caught."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.megakernels import binding as mb
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

BACKEND = "sglang.srt.model_executor.forward_context.get_attn_backend"


class Pool:
    uniform_fp8 = False
    request_window = None

    def __init__(self):
        self.keys = [torch.zeros(4, 8) for _ in range(2)]

    def get_swa_key_buffer_radix(self, layer_id):
        return self.keys[layer_id]

    def get_extra_key_page_size(self, layer_id):
        return 64 + layer_id

    def set_swa_key_buffer_radix_fused_norm_rope(self, **kw):
        return kw


class Step5Pool(Pool):
    """The Step-5 pool's DSA indexer state: per FULL layer, SWA ids rejected."""

    full_layers = (3,)

    def __init__(self):
        super().__init__()
        self.compressed = {3: torch.zeros(5, 8, 1, 256, dtype=torch.uint8)}
        self.key_ring = {3: torch.zeros(4, 8, 1, 256, dtype=torch.bfloat16)}
        self.logits_ring = {3: torch.zeros(4, 8, 1, 256, dtype=torch.bfloat16)}
        self.ring_positions = {3: torch.full((4, 8), -1, dtype=torch.int64)}

    def _full(self, table, layer_id):
        if layer_id not in self.full_layers:
            raise ValueError("Step5 indexer state belongs to FULL attention layers")
        return table[layer_id]

    def get_compressed_index_key_buffer(self, layer_id):
        return self._full(self.compressed, layer_id)

    def get_index_key_ring(self, layer_id):
        return self._full(self.key_ring, layer_id)

    def get_compression_logits_ring(self, layer_id):
        return self._full(self.logits_ring, layer_id)

    def get_index_position_buffer(self, layer_id):
        return self._full(self.ring_positions, layer_id)


STEP5_INDEX_GETTERS = (
    "get_compressed_index_key_buffer",
    "get_index_key_ring",
    "get_compression_logits_ring",
    "get_index_position_buffer",
)


def backend(pool):
    return SimpleNamespace(
        token_to_kv_pool=pool,
        forward_metadata=SimpleNamespace(core_attn_metadata="core"),
        trtllm_attn=False,
        forward=lambda *a, **k: "attention",
        store_cache=lambda **k: None,
        get_swa_out_cache_loc=lambda fb: "swa_loc",
        forward_low_ratio_sources=lambda **k: "sources",
        forward_core_compressor=lambda **k: "compressed",
    )


class TestMegakernelBinding(CustomTestCase):
    def test_the_view_is_the_running_forwards_backend_and_pool(self):
        pool = Pool()
        with patch(BACKEND, return_value=backend(pool)):
            view = mb.MegakernelBinding(model=torch.nn.Module()).forward_view()
        self.assertEqual(view.metadata.core_attn_metadata, "core")
        self.assertIs(view.kv_pool.pool, pool)
        self.assertIs(view.kv_pool.get_swa_key_buffer_radix(1), pool.keys[1])

    def test_a_replaced_source_fails_assert_live_by_name(self):
        model = torch.nn.Module()
        model.engram_hasher = SimpleNamespace(history=torch.zeros(5, 3))
        binding = mb.MegakernelBinding(model=model)
        attn_backend = backend(Pool())
        with patch(BACKEND, return_value=attn_backend):
            binding.forward_view()
            self.assertIs(binding.engram_history(), model.engram_hasher.history)
            binding.assert_live()
            model.engram_hasher.history = torch.zeros(9, 3)
            with self.assertRaisesRegex(RuntimeError, "engram_history"):
                binding.assert_live()
        binding = mb.MegakernelBinding(model=torch.nn.Module())
        with patch(BACKEND, return_value=backend(Pool())):
            binding.forward_view()
        with patch(BACKEND, return_value=backend(Pool())):
            with self.assertRaisesRegex(RuntimeError, "kv_pool"):
                binding.assert_live()

    def test_the_step5_view_asks_the_running_step5_backend(self):
        calls = []
        layer = SimpleNamespace(layer_id=4)
        swa = SimpleNamespace(swa_page_table="pages", cache_seqlens_int32="lens")
        step5_backend = SimpleNamespace(
            token_to_kv_pool=Pool(),
            forward_metadata=SimpleNamespace(indexer_metadata="plan"),
            swa_backend=SimpleNamespace(forward_metadata=swa),
            supports_swa_head_gate=lambda: True,
            supports_dsa_head_gate=lambda: False,
            swa_kv_write_target=lambda l: calls.append(l) or ("k", "v", "loc"),
            get_indexer_metadata=lambda i, fb: (i, fb, "plan"),
        )
        with patch(BACKEND, return_value=step5_backend):
            view = mb.MegakernelBinding(model=torch.nn.Module()).forward_view()
        s5 = view.step5
        self.assertTrue(s5.swa_head_gate)
        self.assertFalse(s5.dsa_head_gate)
        self.assertIs(s5.swa_metadata, swa)
        self.assertEqual(s5.swa_kv_write_target(layer), ("k", "v", "loc"))
        self.assertEqual(calls, [layer])
        self.assertEqual(s5.indexer_metadata(3, "fb"), (3, "fb", "plan"))
        self.assertEqual(
            mb.TOLERATED_STEP5_CALLABLES,
            ("Step5View.swa_kv_write_target", "Step5View.indexer_metadata"),
        )

    def test_another_backend_fails_only_when_the_step5_view_is_read(self):
        with patch(BACKEND, return_value=backend(Pool())):
            view = mb.MegakernelBinding(model=torch.nn.Module()).forward_view()
        s5 = view.step5  # a draft's FA4 backend: built lazily, never read
        with self.assertRaises(AttributeError):
            s5.swa_head_gate

    def test_the_kv_pool_view_reads_the_step5_indexer_state(self):
        pool = Step5Pool()
        with patch(BACKEND, return_value=backend(pool)):
            kv = mb.MegakernelBinding(model=torch.nn.Module()).forward_view().kv_pool
        self.assertIs(kv.get_compressed_index_key_buffer(3), pool.compressed[3])
        self.assertIs(kv.get_index_key_ring(3), pool.key_ring[3])
        self.assertIs(kv.get_compression_logits_ring(3), pool.logits_ring[3])
        self.assertIs(kv.get_index_position_buffer(3), pool.ring_positions[3])
        for name in STEP5_INDEX_GETTERS:
            # an SWA layer has no indexer state: the pool's own error reaches the caller
            with self.assertRaisesRegex(ValueError, "FULL attention layers"):
                getattr(kv, name)(0)
            # tensors under the pool's own names: data, not a tolerated callable
            self.assertNotIn(f"KvPoolView.{name}", mb.TOLERATED_CALLABLES)

    def test_another_pool_fails_only_when_a_step5_index_getter_is_read(self):
        with patch(BACKEND, return_value=backend(Pool())):
            kv = mb.MegakernelBinding(model=torch.nn.Module()).forward_view().kv_pool
        self.assertIs(kv.get_swa_key_buffer_radix(0), kv.pool.keys[0])
        for name in STEP5_INDEX_GETTERS:
            with self.assertRaises(AttributeError):
                getattr(kv, name)(3)


if __name__ == "__main__":
    unittest.main()
