"""Unit tests for ``ModelRunner._disable_symm_mem`` -- CPU only.

Verifies the symmetric-memory toggle helper used by CUDA graph capture:

* The wrapped block sees ``server_args.enable_symm_mem == False``.
* ``is_symmetric_memory_enabled()`` short-circuits to ``False`` inside the
  block, so any ``use_symmetric_memory(...)`` site that would normally
  trigger collective ``ncclMemAlloc`` / ``ncclCommWindowRegister`` calls
  becomes a ``nullcontext()``.
* The original ``enable_symm_mem`` value is restored on normal exit.
* The original value is also restored when the wrapped block raises.
* Nested usage restores the immediately-enclosing value (not just the
  outermost), so wrapping graph capture inside another disabled block is
  safe.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _make_runner(*, enable_symm_mem: bool) -> ModelRunner:
    """Build a stand-in object exposing only the attributes the helper uses.

    ``ModelRunner.__init__`` is heavy (loads weights, sets up dist) and is
    not needed: ``_disable_symm_mem`` only touches
    ``self.server_args.enable_symm_mem``. We bind the unbound method to a
    bare ``SimpleNamespace`` so the test stays GPU-free.
    """
    runner = SimpleNamespace(
        server_args=SimpleNamespace(enable_symm_mem=enable_symm_mem),
    )
    runner._disable_symm_mem = ModelRunner._disable_symm_mem.__get__(runner)
    return runner


class TestDisableSymmMem(unittest.TestCase):
    def test_flips_flag_inside_block(self):
        runner = _make_runner(enable_symm_mem=True)
        with runner._disable_symm_mem():
            self.assertFalse(runner.server_args.enable_symm_mem)
        self.assertTrue(runner.server_args.enable_symm_mem)

    def test_is_symmetric_memory_enabled_returns_false_inside(self):
        # ``is_symmetric_memory_enabled`` reads from the global server-args
        # store via ``get_global_server_args()``. Inside the helper, the
        # local ``server_args.enable_symm_mem`` flag flips to False, which
        # makes every ``use_symmetric_memory(...)`` call site short-circuit
        # to ``nullcontext()`` -- precisely the behavior CUDA-graph capture
        # depends on to avoid the NCCL collective deadlock.
        from sglang.srt.distributed.device_communicators import pynccl_allocator

        runner = _make_runner(enable_symm_mem=True)
        with patch.object(
            pynccl_allocator, "get_global_server_args", return_value=runner.server_args
        ):
            self.assertTrue(pynccl_allocator.is_symmetric_memory_enabled())
            with runner._disable_symm_mem():
                self.assertFalse(pynccl_allocator.is_symmetric_memory_enabled())
            self.assertTrue(pynccl_allocator.is_symmetric_memory_enabled())

    def test_restores_on_exception(self):
        runner = _make_runner(enable_symm_mem=True)
        with self.assertRaises(RuntimeError):
            with runner._disable_symm_mem():
                self.assertFalse(runner.server_args.enable_symm_mem)
                raise RuntimeError("simulated capture-time failure")
        self.assertTrue(
            runner.server_args.enable_symm_mem,
            "enable_symm_mem must be restored when the wrapped block raises",
        )

    def test_no_op_when_already_disabled(self):
        runner = _make_runner(enable_symm_mem=False)
        with runner._disable_symm_mem():
            self.assertFalse(runner.server_args.enable_symm_mem)
        self.assertFalse(runner.server_args.enable_symm_mem)

    def test_nested_restores_enclosing_value(self):
        runner = _make_runner(enable_symm_mem=True)
        with runner._disable_symm_mem():
            self.assertFalse(runner.server_args.enable_symm_mem)
            with runner._disable_symm_mem():
                self.assertFalse(runner.server_args.enable_symm_mem)
            # Exiting the inner block must restore the outer (disabled)
            # value, not the original (enabled) one.
            self.assertFalse(runner.server_args.enable_symm_mem)
        self.assertTrue(runner.server_args.enable_symm_mem)


if __name__ == "__main__":
    unittest.main()
