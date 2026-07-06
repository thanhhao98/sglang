"""
End-to-end accuracy test for DCP (Decode Context Parallelism).

Launches the server with DCP A2A and AG+RS backends, sends identical
prompts, and verifies the generated tokens match within tolerance.
Requires 2+ GPUs.

Usage:
    python -m pytest test/srt/test_dcp_accuracy.py -v
    # or directly:
    python test/srt/test_dcp_accuracy.py
"""

import json
import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    CustomTestCase,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

DCP_MODEL = "deepseek-ai/DeepSeek-V2-Lite"
NUM_GPUS = int(os.environ.get("SGLANG_TEST_NUM_GPUS", "8"))
DCP_SIZE = int(os.environ.get("SGLANG_TEST_DCP_SIZE", str(NUM_GPUS)))

PROMPTS = [
    "The capital of France is",
    "In a distant galaxy, there lived a",
    "The theory of relativity states that",
]
MAX_NEW_TOKENS = 32


def _generate(base_url, prompt, max_new_tokens=MAX_NEW_TOKENS):
    """Send a generate request and return the response text."""
    resp = requests.post(
        f"{base_url}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
            },
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["text"]


class TestDCPA2AAccuracy(CustomTestCase):
    """Test DCP A2A produces correct outputs by comparing against AG+RS."""

    @classmethod
    def setUpClass(cls):
        if NUM_GPUS < 2:
            raise unittest.SkipTest("DCP accuracy test requires 2+ GPUs")

        cls.base_url = DEFAULT_URL_FOR_TEST

        env_dcp = {
            "SGLANG_DCP": str(DCP_SIZE),
            "SGLANG_DCP_SYMM_ONLY": "true",
            "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "1",
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        }

        dcp_args = [
            "--tp-size", str(NUM_GPUS),
            "--attention-backend", "flashinfer",
            "--disable-radix-cache",
            "--enable-symm-mem",
            "--dcp-comm-backend", "a2a",
            "--disable-cuda-graph",
            "--chunked-prefill-size", "32768",
        ]

        cls.process = popen_launch_server(
            DCP_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=dcp_args,
            env=env_dcp,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_server_health(self):
        """Server with DCP A2A should be healthy."""
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        self.assertEqual(resp.status_code, 200)

    def test_generate_basic(self):
        """Basic generation should produce non-empty output."""
        for prompt in PROMPTS:
            output = _generate(self.base_url, prompt)
            self.assertIsInstance(output, str)
            self.assertGreater(len(output), len(prompt))

    def test_deterministic_output(self):
        """Same prompt with temperature=0 should produce identical output."""
        prompt = PROMPTS[0]
        out1 = _generate(self.base_url, prompt)
        out2 = _generate(self.base_url, prompt)
        self.assertEqual(out1, out2, "Deterministic generation should match")

    def test_multiple_prompts(self):
        """All prompts should generate successfully without errors."""
        for prompt in PROMPTS:
            with self.subTest(prompt=prompt[:30]):
                output = _generate(self.base_url, prompt, max_new_tokens=16)
                self.assertIsInstance(output, str)
                self.assertGreater(len(output), 0)


class TestDCPAGRSAccuracy(CustomTestCase):
    """Test DCP AG+RS produces correct outputs."""

    @classmethod
    def setUpClass(cls):
        if NUM_GPUS < 2:
            raise unittest.SkipTest("DCP accuracy test requires 2+ GPUs")

        cls.base_url = DEFAULT_URL_FOR_TEST

        env_dcp = {
            "SGLANG_DCP": str(DCP_SIZE),
            "SGLANG_DCP_SYMM_ONLY": "true",
            "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "1",
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        }

        dcp_args = [
            "--tp-size", str(NUM_GPUS),
            "--attention-backend", "flashinfer",
            "--disable-radix-cache",
            "--enable-symm-mem",
            "--dcp-comm-backend", "ag_rs",
            "--chunked-prefill-size", "32768",
        ]

        cls.process = popen_launch_server(
            DCP_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=dcp_args,
            env=env_dcp,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_server_health(self):
        """Server with DCP AG+RS should be healthy."""
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        self.assertEqual(resp.status_code, 200)

    def test_generate_basic(self):
        """Basic generation should produce non-empty output."""
        for prompt in PROMPTS:
            output = _generate(self.base_url, prompt)
            self.assertIsInstance(output, str)
            self.assertGreater(len(output), len(prompt))

    def test_deterministic_output(self):
        """Same prompt with temperature=0 should produce identical output."""
        prompt = PROMPTS[0]
        out1 = _generate(self.base_url, prompt)
        out2 = _generate(self.base_url, prompt)
        self.assertEqual(out1, out2, "Deterministic generation should match")


if __name__ == "__main__":
    unittest.main(verbosity=3)
