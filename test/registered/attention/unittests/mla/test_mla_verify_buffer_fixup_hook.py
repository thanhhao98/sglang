"""The TRT-LLM MLA family must implement the plan-stream verify fixup hook.

``run_eagle_verify`` calls ``update_verify_buffers_to_fill_after_draft`` on the
target backend unconditionally under ``SGLANG_ENABLE_OVERLAP_PLAN_STREAM``. The
base class raises ``NotImplementedError`` by design, so a backend that reaches
that call without an override kills EAGLE/EAGLE3 on its first verify batch.

For this family the correct body is a no-op: it consumes no mask (``verify_mask``
is created with ``is_read=False``) and its verify metadata is a function of
``seq_lens`` / ``req_pool_indices`` / ``req_to_token`` and the config-time
``num_draft_tokens``, never of the sampled draft tokens. What must be pinned is
that the override *exists*, that both call shapes in the tree work, that the
subclasses inherit it, and that the base still fails loudly for everyone else.

Pure host-side attribute/dispatch logic — CPU tensors, no CUDA required.
"""

import unittest

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    TRTLLMMLADecodeMetadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-large")

_STALE = -555
_BS = 4

_HOOK = "update_verify_buffers_to_fill_after_draft"

# The subclasses pull in kernel modules that may be absent in a given build.
# Their inheritance is the whole point of putting the override on the parent, so
# skip rather than drop the assertion.
_SUBCLASSES = {}
try:
    from sglang.srt.layers.attention.cutedsl_mla_backend import CuteDslMLABackend

    _SUBCLASSES["cutedsl_mla"] = CuteDslMLABackend
except ImportError:  # pragma: no cover - build-dependent
    pass
try:
    from sglang.srt.layers.attention.tokenspeed_mla_backend import TokenspeedMLABackend

    _SUBCLASSES["tokenspeed_mla"] = TokenspeedMLABackend
except ImportError:  # pragma: no cover - build-dependent
    pass


def _make_backend() -> TRTLLMMLABackend:
    """Bare backend carrying only the verify metadata the hook could touch."""
    backend = object.__new__(TRTLLMMLABackend)
    metadata = TRTLLMMLADecodeMetadata()
    metadata.seq_lens_k = torch.full((_BS,), _STALE, dtype=torch.int32)
    metadata.global_seq_lens_k = torch.full((_BS,), _STALE, dtype=torch.int32)
    metadata.block_kv_indices = torch.full((_BS, 8), _STALE, dtype=torch.int32)
    backend.decode_cuda_graph_metadata = {_BS: metadata}
    backend.forward_decode_metadata = metadata
    backend._verify_mask = None
    return backend


class _StubSpecInput:
    """Stand-in for EagleVerifyInput: the hook must not read anything off it."""

    def __getattr__(self, name):
        if name.startswith("__"):  # let unittest/copy protocols probe freely
            raise AttributeError(name)
        raise AssertionError(f"the no-op hook must not read spec_info.{name}")


class _RecordingChild:
    def __init__(self):
        self.calls = []

    def update_verify_buffers_to_fill_after_draft(self, spec_info, cuda_graph_bs):
        self.calls.append((spec_info, cuda_graph_bs))


class TestMLAVerifyBufferFixupHook(CustomTestCase):
    def test_override_exists_on_the_family_root(self):
        # The regression guard: without an override the base raises and EAGLE
        # dies on the first verify batch under the overlap plan stream.
        self.assertIsNot(
            getattr(TRTLLMMLABackend, _HOOK),
            getattr(AttentionBackend, _HOOK),
            "TRTLLMMLABackend must override the plan-stream verify fixup hook",
        )

    def test_noop_leaves_verify_metadata_untouched(self):
        backend = _make_backend()
        metadata = backend.decode_cuda_graph_metadata[_BS]

        backend.update_verify_buffers_to_fill_after_draft(_StubSpecInput(), _BS)

        for name in ("seq_lens_k", "global_seq_lens_k", "block_kv_indices"):
            self.assertTrue(
                (getattr(metadata, name) == _STALE).all(),
                f"{name} must not be rewritten by the no-op hook",
            )
        self.assertIs(backend.decode_cuda_graph_metadata[_BS], metadata)

    def test_accepts_positional_and_keyword_call_shapes(self):
        # HybridAttnBackend calls positionally; HybridLinearAttnBackend calls
        # with keywords. Both must keep working.
        backend = _make_backend()
        backend.update_verify_buffers_to_fill_after_draft(_StubSpecInput(), _BS)
        backend.update_verify_buffers_to_fill_after_draft(
            spec_info=_StubSpecInput(), cuda_graph_bs=_BS
        )

    def test_eager_path_none_cuda_graph_bs_is_accepted(self):
        backend = _make_backend()
        backend.update_verify_buffers_to_fill_after_draft(_StubSpecInput(), None)

    def test_subclasses_inherit_the_override(self):
        if not _SUBCLASSES:
            self.skipTest("neither cutedsl_mla nor tokenspeed_mla is importable")
        for name, cls in _SUBCLASSES.items():
            with self.subTest(backend=name):
                self.assertTrue(issubclass(cls, TRTLLMMLABackend))
                self.assertIs(
                    getattr(cls, _HOOK),
                    getattr(TRTLLMMLABackend, _HOOK),
                    f"{name} must inherit the family no-op, not shadow it",
                )

    def test_hybrid_delegates_to_the_target_verify_child(self):
        # The reported repro routes verify through HybridAttnBackend with
        # --speculative-attention-mode decode, i.e. to the decode child.
        for spec_attn_is_decode, expected in ((True, "decode"), (False, "prefill")):
            with self.subTest(spec_attn_is_decode=spec_attn_is_decode):
                hybrid = object.__new__(HybridAttnBackend)
                children = {"decode": _RecordingChild(), "prefill": _RecordingChild()}
                hybrid.decode_backend = children["decode"]
                hybrid.prefill_backend = children["prefill"]
                hybrid.spec_attn_is_decode = spec_attn_is_decode

                self.assertIs(
                    hybrid._select_backend(ForwardMode.TARGET_VERIFY),
                    children[expected],
                )

                spec_info = _StubSpecInput()
                hybrid.update_verify_buffers_to_fill_after_draft(spec_info, _BS)

                self.assertEqual(children[expected].calls, [(spec_info, _BS)])
                other = "prefill" if expected == "decode" else "decode"
                self.assertEqual(children[other].calls, [])

    def test_base_still_raises_for_backends_without_an_override(self):
        # A blanket no-op would be wrong for backends whose verify plan packs a
        # custom mask; the fail-loud default must stay.
        class _Unpatched(AttentionBackend):
            pass

        backend = object.__new__(_Unpatched)
        with self.assertRaises(NotImplementedError):
            backend.update_verify_buffers_to_fill_after_draft(None, None)


if __name__ == "__main__":
    unittest.main()
