"""Tests for ``use_symmetric_memory(..., allow_under_graph_capture=...)``."""

import contextlib
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.distributed.device_communicators import pynccl_allocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestSymmMemGraphCaptureGate(CustomTestCase):
    def test_capture_without_allow_short_circuits_to_nullcontext(self):
        """Under graph capture, symm is off unless explicitly allowed."""
        mock_group = MagicMock()
        mock_group.world_size = 2

        mock_dm = MagicMock()
        mock_dm.is_current_stream_capturing = MagicMock(return_value=True)

        with (
            patch.object(
                pynccl_allocator,
                "is_symmetric_memory_enabled",
                return_value=True,
            ),
            patch.object(torch, "get_device_module", return_value=mock_dm),
        ):
            ctx = pynccl_allocator.use_symmetric_memory(mock_group)
            self.assertIsInstance(ctx, contextlib.nullcontext)

    def test_capture_with_allow_still_short_circuits_when_symm_disabled(self):
        """``allow_under_graph_capture`` does not bypass global symm disable."""
        mock_group = MagicMock()
        mock_group.world_size = 2

        mock_dm = MagicMock()
        mock_dm.is_current_stream_capturing = MagicMock(return_value=True)

        with (
            patch.object(
                pynccl_allocator,
                "is_symmetric_memory_enabled",
                return_value=False,
            ),
            patch.object(torch, "get_device_module", return_value=mock_dm),
        ):
            ctx = pynccl_allocator.use_symmetric_memory(
                mock_group, allow_under_graph_capture=True
            )
            self.assertIsInstance(ctx, contextlib.nullcontext)


if __name__ == "__main__":
    unittest.main()
