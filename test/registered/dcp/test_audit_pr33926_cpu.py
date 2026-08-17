"""Audit pin for PR #33926, finding 4 (CPU-only, no GPU / no flashinfer call).

Contract under test
-------------------
``TRTLLMMLABackend._run_decode_kernel`` (the decode-kernel hook) honors
``return_lse`` on the PR head: with ``return_lse=True`` it returns
``(out, lse)``, with ``False`` a bare tensor (trtllm_mla_backend.py, the
``return_lse=return_lse`` forward into the flashinfer call).

``CuteDslMLABackend._run_decode_kernel`` overrides the hook and, on the
``cp_world <= 1`` branch, delegates to the base implementation:

    if cp_world <= 1:
        return super()._run_decode_kernel(
            query, kv_cache, block_tables, seq_lens, max_seq_len, layer
        )

The finding: this delegation drops ``return_lse`` (and every other keyword),
so a caller invoking the hook on a cute-dsl backend with
``cp_world=1, return_lse=True`` silently gets a bare tensor where the hook
contract promises ``(out, lse)``. At the merge base the base hook refused
``return_lse=True`` loudly (``NotImplementedError``), so the base could never
be *silently* wrong about the LSE; the PR made the base honor the kwarg,
which turns the dropped kwarg into a silent contract violation.

``test_cp_world_one_delegation_forwards_return_lse`` pins the CORRECT
contract and is therefore EXPECTED TO FAIL on the unfixed PR head
(197f35c25f) -- that failure is the empirical confirmation of the finding.

No GPU, no flashinfer: the base hook is replaced by a recorder before the
call, so nothing past the delegation line executes. When optional heavyweight
deps are absent (local macOS run), minimal import stubs are installed first;
inside the CI/cluster container the stubs are skipped because the real
modules resolve.

Usage:
    python -m pytest test_audit_pr33926_cpu.py -v
    python test_audit_pr33926_cpu.py
"""

import importlib.util
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch


def _install_import_stubs_if_needed() -> None:
    """Make ``sglang.srt.layers.attention.cutedsl_mla_backend`` importable on
    a box without the CUDA serving deps. Every stub is conditional on the real
    module being absent, so a full container environment is untouched."""
    if importlib.util.find_spec("xgrammar") is None:
        import pydantic

        xg = types.ModuleType("xgrammar")
        xg.__path__ = []  # mark as package so submodule import resolves
        st = types.ModuleType("xgrammar.structural_tag")
        for name in (
            "StructuralTag",
            "StructuralTagItem",
            "AnyTextFormat",
            "AnyTokensFormat",
            "ConstStringFormat",
            "ExcludeTokenFormat",
            "Format",
            "JSONSchemaFormat",
            "OptionalFormat",
            "OrFormat",
            "RegexFormat",
            "SequenceFormat",
            "StarFormat",
            "TagFormat",
            "TagsWithSeparatorFormat",
            "TokenFormat",
            "TriggeredTagsFormat",
            "CompiledGrammar",
            "GrammarCompiler",
            "GrammarMatcher",
            "TokenizerInfo",
        ):
            # pydantic BaseModel subclasses so downstream pydantic schemas
            # built over these types still generate.
            cls = type(
                name,
                (pydantic.BaseModel,),
                {"model_config": pydantic.ConfigDict(extra="allow")},
            )
            setattr(xg, name, cls)
            setattr(st, name, cls)
        xg.bitmask_dtype = None
        xg.get_bitmask_shape = lambda *a, **k: (0, 0)
        xg.get_model_structural_tag = lambda *a, **k: None
        xg.apply_token_bitmask_inplace = lambda *a, **k: None
        xg.structural_tag = st
        sys.modules["xgrammar"] = xg
        sys.modules["xgrammar.structural_tag"] = st

    if importlib.util.find_spec("triton") is None:
        # torch 2.13 without triton cannot import torch._inductor (module-level
        # `CompiledKernel | ...` union over a placeholder), and sglang has
        # module-level @torch.compile decorators on the import path. Neutralize
        # the decorator; nothing in this test executes a compiled region.
        import torch

        def _no_compile(fn=None, **_kwargs):
            if callable(fn):
                return fn
            return lambda f: f

        torch.compile = _no_compile


_install_import_stubs_if_needed()

try:
    import torch

    from sglang.srt.layers.attention.cutedsl_mla_backend import CuteDslMLABackend
    from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend
except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent skip
    import pytest

    pytest.skip(
        f"cutedsl_mla_backend not importable in this environment: {exc}",
        allow_module_level=True,
    )


def _make_hook_args():
    """Minimal CPU stand-ins for the positional hook arguments. The
    ``cp_world <= 1`` branch forwards them untouched, so shape is irrelevant."""
    query = torch.zeros(2, 1, 4)
    kv_cache = torch.zeros(1, 1, 8, 4)
    block_tables = torch.zeros(2, 1, dtype=torch.int32)
    seq_lens = torch.tensor([3, 5], dtype=torch.int32)
    max_seq_len = 5
    # getattr-tolerant stand-in; the delegation must not require a real layer.
    layer = SimpleNamespace(layer_id=0, scaling=1.0, k_scale_float=None)
    return query, kv_cache, block_tables, seq_lens, max_seq_len, layer


class TestCuteDslDelegationForwardsReturnLse(unittest.TestCase):
    def _call_cutedsl_hook_with_base_recorder(self, **hook_kwargs):
        """Invoke the cute-dsl override with the base hook replaced by a
        recorder. Returns (recorded_call_or_None, hook_result)."""
        override = CuteDslMLABackend.__dict__.get("_run_decode_kernel")
        if override is None:
            self.skipTest(
                "CuteDslMLABackend no longer overrides _run_decode_kernel; "
                "the base hook serves the call directly and the contract "
                "holds trivially"
            )

        recorded = {}

        def recorder(_self, *args, **kwargs):
            recorded["args"] = args
            recorded["kwargs"] = kwargs
            if kwargs.get("return_lse", False):
                return ("SENTINEL_OUT", "SENTINEL_LSE")
            return "SENTINEL_OUT"

        backend = object.__new__(CuteDslMLABackend)  # no __init__: no GPU
        args = _make_hook_args()
        with patch.object(TRTLLMMLABackend, "_run_decode_kernel", recorder):
            result = override(backend, *args, **hook_kwargs)
        return (recorded if recorded else None), result

    def test_cp_world_one_delegation_forwards_return_lse(self):
        """Finding 4 pin: EXPECTED TO FAIL on PR head 197f35c25f.

        With cp_world=1 and return_lse=True the override must forward
        return_lse to the base hook (which now honors it) and hand back the
        (out, lse) pair. The PR head's delegation drops the kwarg, so the
        recorder sees no return_lse and the caller gets a bare tensor.
        """
        recorded, result = self._call_cutedsl_hook_with_base_recorder(
            cp_world=1, return_lse=True
        )
        self.assertIsNotNone(
            recorded, "cp_world=1 call never reached the base decode hook"
        )
        forwarded = recorded["kwargs"].get("return_lse", None)
        self.assertIs(
            forwarded,
            True,
            "CuteDslMLABackend._run_decode_kernel(cp_world=1, return_lse=True) "
            f"delegated to the base hook with return_lse={forwarded!r}; the "
            "kwarg was dropped, so the base returns a bare tensor where the "
            "hook contract promises (out, lse)",
        )
        self.assertEqual(
            result,
            ("SENTINEL_OUT", "SENTINEL_LSE"),
            "hook result is not the (out, lse) pair the return_lse=True "
            f"contract promises; got {result!r}",
        )

    def test_cp_world_one_delegation_reaches_base_hook(self):
        """Harness sanity (passes on PR head): the cp_world<=1 branch does
        delegate to the base hook, with the positional arguments intact."""
        recorded, result = self._call_cutedsl_hook_with_base_recorder(cp_world=1)
        self.assertIsNotNone(recorded)
        self.assertEqual(len(recorded["args"]), 6)
        self.assertEqual(result, "SENTINEL_OUT")

    def test_dcp_branch_requires_causal_seqs(self):
        """Harness sanity (passes on PR head): the DCP branch raises before
        touching flashinfer when the global causal bound is missing."""
        with self.assertRaises(ValueError):
            self._call_cutedsl_hook_with_base_recorder(
                cp_world=2, cp_rank=0, return_lse=True
            )


if __name__ == "__main__":
    unittest.main()
