"""CPU unit test for the decode-context-parallel (DCP) strategy contract (P2).

Pins the DecodeContextParallelStrategy seam added in Phase 2:

* the prefill strategies (interleave/zigzag) do NOT implement the decode contract
  (``supports_decode()`` is False and the decode methods raise NotImplementedError,
  so adding the contract to the ABC left them untouched);
* ``DecodeContextParallelStrategy`` reports ``supports_decode()`` and keeps the
  ``INTERLEAVE`` kind (DCP's ``pos % N == rank`` owner rule IS the interleave layout);
* its delegating methods are behavior-preserving wrappers over the ``layers/cp/dcp``
  free functions (``local_decode_kv_lens`` == ``get_dcp_lens``; ``merge_decode_attention``
  dispatches mha/mla and short-circuits on a world_size==1 group);
* ``can_apply`` fires on decode (not extend), unlike the interleave parent.

Usage:
    python -m pytest test_dcp_strategy_unit.py -v
    python test_dcp_strategy_unit.py
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.cp.base import ContextParallelStrategyKind
from sglang.srt.layers.cp.dcp.layout import get_dcp_lens
from sglang.srt.layers.cp.dcp.strategy import DecodeContextParallelStrategy
from sglang.srt.layers.cp.interleave import InterleaveCPStrategy
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

DCP_SIZES = [1, 2, 3, 4, 8]
LENS = list(range(0, 41))

# Decode-contract methods that must raise on non-decode strategies, with dummy
# args so argument-binding succeeds and the NotImplementedError body is what fires
# (None args are never dereferenced — the ABC body raises before using them).
_DECODE_CALLS = [
    ("decode_cp_size", (), {}),
    ("decode_cp_rank", (), {}),
    ("decode_cp_group", (), {}),
    ("local_decode_kv_lens", (None, 1, 0), {}),
    ("update_local_decode_kv_lens", (None,), {}),
    ("shard_decode_kv_indices", (None,), {}),
    ("build_decode_metadata", (), {}),
    ("plan_decode_metadata", (), {}),
    ("gather_decode_query", (None, None), {}),
    ("merge_decode_attention", (None, None, None), {"backend": "mha"}),
]


def _fake_batch(is_decode: bool):
    mode = SimpleNamespace(
        is_decode=lambda: is_decode,
        is_extend=lambda: not is_decode,
    )
    return SimpleNamespace(forward_mode=mode)


class _FakeGroup:
    """A single-rank group: the merge/gather ops short-circuit on world_size==1."""

    world_size = 1
    rank_in_group = 0


class TestDecodeStrategyContract(unittest.TestCase):
    def test_prefill_strategies_do_not_support_decode(self):
        for strategy in (InterleaveCPStrategy(cp_size=2), ZigzagCPStrategy(cp_size=2)):
            self.assertFalse(strategy.supports_decode())
            # Every decode-contract method must raise NotImplementedError (the ABC
            # default), not silently no-op — that is what keeps them "untouched".
            for name, args, kwargs in _DECODE_CALLS:
                with self.assertRaises(
                    NotImplementedError, msg=f"{strategy.name}.{name}"
                ):
                    getattr(strategy, name)(*args, **kwargs)

    def test_decode_strategy_identity(self):
        strategy = DecodeContextParallelStrategy(cp_size=4)
        self.assertTrue(strategy.supports_decode())
        self.assertEqual(strategy.kind, ContextParallelStrategyKind.INTERLEAVE)
        self.assertEqual(strategy.name, "decode_context_parallel")
        self.assertEqual(strategy.cp_size, 4)

    def test_local_decode_kv_lens_matches_free_function(self):
        strategy = DecodeContextParallelStrategy(cp_size=2)
        for n in DCP_SIZES:
            for rank in range(n):
                lens = torch.tensor(LENS, dtype=torch.int32)
                got = strategy.local_decode_kv_lens(lens, n, rank)
                ref = get_dcp_lens(lens, n, rank)
                self.assertTrue(
                    torch.equal(got, ref),
                    f"delegation mismatch at n={n}, rank={rank}",
                )

    def test_can_apply_fires_on_decode_only(self):
        strategy = DecodeContextParallelStrategy(cp_size=2)
        self.assertTrue(strategy.can_apply(1, _fake_batch(is_decode=True)))
        self.assertFalse(strategy.can_apply(1, _fake_batch(is_decode=False)))
        # cp_size <= 1 disables DCP regardless of mode.
        self.assertFalse(
            DecodeContextParallelStrategy(cp_size=1).can_apply(
                1, _fake_batch(is_decode=True)
            )
        )

    def test_merge_decode_attention_dispatch(self):
        strategy = DecodeContextParallelStrategy(cp_size=2)
        out = torch.randn(3, 2, 4)
        lse = torch.randn(3, 2)
        group = _FakeGroup()
        # world_size==1 short-circuit: both backends return the input unchanged.
        mha = strategy.merge_decode_attention(out, lse, group, backend="mha")
        self.assertTrue(torch.equal(mha, out))
        mla = strategy.merge_decode_attention(out, lse, group, backend="mla")
        self.assertTrue(torch.equal(mla, out))
        # return_lse path (mha) yields the (out, lse) tuple on the single-rank group.
        mha_o, mha_lse = strategy.merge_decode_attention(
            out, lse, group, backend="mha", return_lse=True
        )
        self.assertTrue(torch.equal(mha_o, out) and torch.equal(mha_lse, lse))
        with self.assertRaises(ValueError):
            strategy.merge_decode_attention(out, lse, group, backend="bogus")


if __name__ == "__main__":
    unittest.main()
