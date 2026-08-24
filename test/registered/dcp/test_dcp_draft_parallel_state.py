"""Test the DCP-flat draft ParallelState and the layer-owned DCP dispatch flag.

Drafts are TP-sharded and never split the token dimension, so a draft worker
freezes ``ps.dcp_flat()`` at construction (TpModelWorker chokepoint) and the
DeepSeek MLA dispatch predicate keys on the layer's own ``dcp_enabled`` — no
process-global state is scoped or overridden anywhere on the draft path.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, stage="base-a")


def _mode(decode=False, target_verify=False, extend=False):
    return SimpleNamespace(
        is_decode=lambda: decode,
        is_target_verify=lambda: target_verify,
        is_extend=lambda: extend,
    )


class TestDraftParallelState(CustomTestCase):
    def test_dcp_flat_flattens_only_dcp(self):
        ps = ParallelState.trivial(
            tp_rank=5,
            tp_size=8,
            attn_tp_rank=5,
            attn_tp_size=8,
            attn_dcp_rank=3,
            attn_dcp_size=8,
        )
        flat = ps.dcp_flat()
        self.assertEqual(flat.attn_dcp_size, 1)
        self.assertEqual(flat.attn_dcp_rank, 0)
        # Every non-DCP field is preserved: the draft keeps the target's TP world.
        self.assertEqual(flat.tp_rank, 5)
        self.assertEqual(flat.tp_size, 8)
        self.assertEqual(flat.attn_tp_rank, 5)
        self.assertEqual(flat.attn_tp_size, 8)

    def test_dcp_flat_is_idempotent_and_pure(self):
        ps = ParallelState.trivial(attn_dcp_rank=1, attn_dcp_size=4)
        flat = ps.dcp_flat()
        self.assertEqual(flat.dcp_flat(), flat)
        # The source state is untouched (frozen dataclass, new instance).
        self.assertEqual(ps.attn_dcp_size, 4)
        self.assertEqual(ps.attn_dcp_rank, 1)

    def test_dcp_flat_without_dcp_is_identity(self):
        ps = ParallelState.trivial(tp_rank=2, tp_size=4)
        self.assertEqual(ps.dcp_flat(), ps)


class TestLayerOwnedDcpDispatch(CustomTestCase):
    """is_dcp_mla_decode_phase keys on the layer's flag, never the global."""

    def _predicate(self, attn_dcp_enabled, mode):
        from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
            is_dcp_mla_decode_phase,
        )

        attn = SimpleNamespace(dcp_enabled=attn_dcp_enabled)
        return is_dcp_mla_decode_phase(attn, SimpleNamespace(forward_mode=mode))

    def test_draft_layer_never_dispatches_dcp(self):
        """A DCP-off layer (nextn/MTP draft) resolves False even while the
        process-global DCP state is live — the global must not leak back in."""
        from sglang.srt import runtime_context as rc

        with rc.get_parallel().override(dcp_enabled=True, dcp_size=8, dcp_rank=3):
            for mode in (_mode(decode=True), _mode(target_verify=True)):
                self.assertFalse(self._predicate(False, mode))

    def test_target_layer_dispatches_by_mode(self):
        self.assertTrue(self._predicate(True, _mode(decode=True)))
        self.assertTrue(self._predicate(True, _mode(target_verify=True)))
        self.assertFalse(self._predicate(True, _mode(extend=True)))

    def test_dcp_off_layer_ignores_mode(self):
        for mode in (_mode(decode=True), _mode(target_verify=True), _mode()):
            self.assertFalse(self._predicate(False, mode))


if __name__ == "__main__":
    unittest.main()
