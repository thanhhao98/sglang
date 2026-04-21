"""Unit tests for ``ModelRunner._disable_symm_mem``.

The helper is a small ``@contextmanager`` whose only job is to gate
``use_symmetric_memory()`` short-circuiting through the global server-args
store, so that NCCL collective ``ncclMemAlloc`` / ``ncclCommWindowRegister``
calls are skipped during CUDA graph capture.

Two behaviors carry real correctness risk and are covered here:

1. **End-to-end short-circuit**: while the helper is active,
   ``is_symmetric_memory_enabled()`` (the predicate ``use_symmetric_memory``
   reads) must return ``False``. If this contract breaks, the deadlock the
   helper exists to prevent comes back.
2. **Restore on exception**: if an exception escapes the wrapped block, the
   original ``enable_symm_mem`` value must be restored. If this breaks,
   live serving silently loses symm-mem after any capture-time failure.

Pure-Python ``try/finally`` save/restore semantics are not re-tested
(the SGLang test skill: "skip tests that just verify Python itself works").
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.distributed.device_communicators import pynccl_allocator
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _make_runner(enable_symm_mem: bool):
    """Bind the unbound ``_disable_symm_mem`` onto a stand-in object.

    The helper only touches ``self.server_args.enable_symm_mem`` -- no GPU,
    no dist init, no model load -- so we skip the heavy
    ``ModelRunner.__init__`` and attach the method to a ``SimpleNamespace``
    that exposes just the attribute the helper reads.
    """
    runner = SimpleNamespace(
        server_args=SimpleNamespace(enable_symm_mem=enable_symm_mem),
    )
    runner._disable_symm_mem = ModelRunner._disable_symm_mem.__get__(runner)
    return runner


class TestDisableSymmMem(CustomTestCase):
    def test_short_circuits_is_symmetric_memory_enabled(self):
        """The end-to-end contract that prevents the deadlock.

        ``use_symmetric_memory(group)`` returns ``nullcontext()`` when
        ``is_symmetric_memory_enabled()`` is ``False``. That predicate
        reads from ``get_global_server_args().enable_symm_mem``. We patch
        the global getter to point at our stand-in so the assertions
        exercise the same code path the live server takes.
        """
        runner = _make_runner(enable_symm_mem=True)
        with patch.object(
            pynccl_allocator,
            "get_global_server_args",
            return_value=runner.server_args,
        ):
            self.assertTrue(pynccl_allocator.is_symmetric_memory_enabled())
            with runner._disable_symm_mem():
                self.assertFalse(pynccl_allocator.is_symmetric_memory_enabled())
            self.assertTrue(pynccl_allocator.is_symmetric_memory_enabled())

    def test_restores_enable_symm_mem_when_block_raises(self):
        """If the wrapped block raises, the flag must still be restored.

        Without this, a transient capture-time failure would silently
        disable symm-mem for the whole serving session.
        """
        runner = _make_runner(enable_symm_mem=True)
        with self.assertRaises(RuntimeError):
            with runner._disable_symm_mem():
                self.assertFalse(runner.server_args.enable_symm_mem)
                raise RuntimeError("simulated capture-time failure")
        self.assertTrue(runner.server_args.enable_symm_mem)


if __name__ == "__main__":
    unittest.main()
