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

import ast
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.distributed.device_communicators import pynccl_allocator
from sglang.srt.model_executor import cuda_graph_runner, model_runner
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


class TestDisableSymmMemScope(CustomTestCase):
    """Guardrails against regressing the wrapper scope.

    The whole point of the narrow-scope fix is that ``_disable_symm_mem``
    wraps ONLY the warmup forwards, not the actual stream-captured
    forward. If a future refactor puts the captured forward back under
    ``_disable_symm_mem``, the captured graph bakes in the non-symm-mem
    path and symm-mem is silently dead at replay even though the server
    no longer hangs.

    These tests parse the relevant source functions with ``ast`` and
    fail loudly on that regression. AST beats ``inspect.getsource``
    substring matching: it ignores comments / whitespace and makes the
    "A is inside the ``with`` block, B is not" question precise.
    """

    @staticmethod
    def _block_contains(body, predicate):
        for node in ast.walk(ast.Module(body=list(body), type_ignores=[])):
            if predicate(node):
                return True
        return False

    @staticmethod
    def _is_disable_symm_mem_with(node):
        if not isinstance(node, ast.With):
            return False
        for item in node.items:
            call = item.context_expr
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if isinstance(func, ast.Attribute) and func.attr == "_disable_symm_mem":
                return True
        return False

    @staticmethod
    def _calls_attr(name):
        def _pred(node):
            return (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == name
            )

        return _pred

    def _parse_fn(self, module, qualname):
        src = inspect.getsource(module)
        tree = ast.parse(src)
        parts = qualname.split(".")
        node = tree
        for part in parts:
            found = None
            for child in ast.walk(node):
                if (
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == part
                ):
                    found = child
                    break
                if isinstance(child, ast.ClassDef) and child.name == part:
                    found = child
                    break
            self.assertIsNotNone(
                found, f"could not locate {part!r} in {module.__name__}"
            )
            node = found
        return node

    def test_decode_capture_wraps_only_warmups(self):
        """``_capture_graph`` must be called OUTSIDE ``_disable_symm_mem``.

        If the fix regresses and the capture call moves back inside the
        ``with self.model_runner._disable_symm_mem():`` block, the
        captured CUDA graph will not exercise symm-mem at replay.
        """
        fn = self._parse_fn(
            cuda_graph_runner, "CudaGraphRunner.capture_one_batch_size"
        )
        # Confirm the warmup wrapper exists at all — guards against the
        # opposite regression (deleting the wrapper and reintroducing the
        # deadlock).
        self.assertTrue(
            self._block_contains(fn.body, self._is_disable_symm_mem_with),
            "capture_one_batch_size must wrap warmups in _disable_symm_mem",
        )
        # And confirm _capture_graph is NOT inside any such wrapper.
        for node in ast.walk(fn):
            if not self._is_disable_symm_mem_with(node):
                continue
            self.assertFalse(
                self._block_contains(node.body, self._calls_attr("_capture_graph")),
                "_capture_graph must be called outside _disable_symm_mem",
            )

    def test_init_device_graphs_does_not_wrap_runner_ctor(self):
        """The outer wrapper around ``CudaGraphRunner(self)`` must stay removed.

        That wrapper is what made the captured graph non-symm-mem in the
        first place; it lives *inside* the runner now, scoped to warmups
        only.
        """
        fn = self._parse_fn(model_runner, "ModelRunner.init_device_graphs")
        for node in ast.walk(fn):
            if not self._is_disable_symm_mem_with(node):
                continue
            # If a _disable_symm_mem block exists here, it must not contain
            # any CudaGraphRunner / platform GraphRunnerCls construction.
            def _constructs_graph_runner(n):
                return (
                    isinstance(n, ast.Call)
                    and isinstance(n.func, (ast.Name, ast.Attribute))
                    and any(
                        s in ast.dump(n.func)
                        for s in ("GraphRunner", "CudaGraphRunner")
                    )
                )

            self.assertFalse(
                self._block_contains(node.body, _constructs_graph_runner),
                "init_device_graphs must not wrap graph runner construction"
                " in _disable_symm_mem",
            )


if __name__ == "__main__":
    unittest.main()
