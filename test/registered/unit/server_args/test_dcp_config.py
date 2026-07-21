"""Unit tests for DCP (Decode Context Parallelism) server args configuration.

Covers the ``--dcp-comm-backend`` field ({ag_rs, a2a, fi_a2a}) and its
validation in ``ServerArgs._handle_dcp_validation``:
  - a2a / fi_a2a require --dcp-size > 1
  - fi_a2a requires a CUDA platform (the authoritative MNNVL fabric probe runs
    later, at model-runner init)
  - dcp>1 requires CUDA or HIP (base behavior from the merged DCP PR)
  - CUDA speculative DCP is restricted to the validated tokenspeed_mla
    EAGLE/EAGLE3 chain configuration

Tests construct with safe defaults (dcp_size=1) then mutate the fields and call
``_handle_dcp_validation`` directly, so construction never trips the platform
gate; is_cuda / is_hip are patched per-test to pin the platform deterministically
(these are CPU-CI tests, where the real is_cuda() is False).
"""

import dataclasses
import unittest
from unittest.mock import patch

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_mock_device = patch("sglang.srt.server_args.get_device", return_value="cuda")
_mock_device.start()


class TestDCPFieldDefaults(CustomTestCase):
    """Verify DCP-related dataclass fields exist with correct defaults."""

    def test_dcp_size_field_exists(self):
        fields = {f.name for f in dataclasses.fields(ServerArgs)}
        self.assertIn("dcp_size", fields)

    def test_dcp_comm_backend_field_exists(self):
        fields = {f.name for f in dataclasses.fields(ServerArgs)}
        self.assertIn("dcp_comm_backend", fields)

    def test_dcp_size_default(self):
        self.assertEqual(ServerArgs.dcp_size, 1)

    def test_dcp_comm_backend_default(self):
        self.assertEqual(ServerArgs.dcp_comm_backend, "ag_rs")


class TestDCPCommBackendValidation(CustomTestCase):
    """Verify ``_handle_dcp_validation`` accepts/rejects the right combos."""

    @staticmethod
    def _make_args(dcp_size, dcp_comm_backend):
        # Construct with safe defaults (dcp_size=1) so __post_init__ never trips
        # the dcp>1 platform gate, then set the fields under test.
        args = ServerArgs(model_path="dummy")
        args.dcp_size = dcp_size
        args.dcp_comm_backend = dcp_comm_backend
        return args

    def test_a2a_requires_dcp_size_gt_1(self):
        args = self._make_args(dcp_size=1, dcp_comm_backend="a2a")
        with self.assertRaises(ValueError):
            args._handle_dcp_validation()

    def test_fi_a2a_requires_dcp_size_gt_1(self):
        args = self._make_args(dcp_size=1, dcp_comm_backend="fi_a2a")
        with self.assertRaises(ValueError):
            args._handle_dcp_validation()

    def test_dcp_size_must_be_positive(self):
        args = self._make_args(dcp_size=0, dcp_comm_backend="ag_rs")
        with self.assertRaisesRegex(ValueError, "must be >= 1"):
            args._handle_dcp_validation()

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    @patch("sglang.srt.server_args.is_cuda", return_value=True)
    def test_a2a_with_dcp_size_2_on_cuda_passes(self, *_):
        args = self._make_args(dcp_size=2, dcp_comm_backend="a2a")
        args._handle_dcp_validation()  # no raise
        self.assertEqual(args.dcp_comm_backend, "a2a")

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    @patch("sglang.srt.server_args.is_cuda", return_value=True)
    def test_fi_a2a_with_dcp_size_2_on_cuda_passes_server_args(self, *_):
        # server_args accepts fi_a2a on CUDA; the MNNVL fabric probe is deferred
        # to model-runner init (init_fi_a2a_workspace).
        args = self._make_args(dcp_size=2, dcp_comm_backend="fi_a2a")
        args._handle_dcp_validation()  # no raise
        self.assertEqual(args.dcp_comm_backend, "fi_a2a")

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    @patch("sglang.srt.server_args.is_cuda", return_value=False)
    def test_fi_a2a_on_non_cuda_raises(self, *_):
        args = self._make_args(dcp_size=2, dcp_comm_backend="fi_a2a")
        with self.assertRaises(ValueError):
            args._handle_dcp_validation()

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    @patch("sglang.srt.server_args.is_cuda", return_value=True)
    def test_ag_rs_with_dcp_size_8_on_cuda_passes(self, *_):
        args = self._make_args(dcp_size=8, dcp_comm_backend="ag_rs")
        args._handle_dcp_validation()  # no raise
        self.assertEqual(args.dcp_size, 8)


class TestDCPSpeculativeAllowlist(CustomTestCase):
    """Verify the narrow CUDA speculative-DCP server-argument allowlist."""

    @staticmethod
    def _make_args(**overrides):
        args = ServerArgs(model_path="dummy")
        values = {
            "dcp_size": 8,
            "dcp_comm_backend": "a2a",
            "speculative_algorithm": "EAGLE3",
            "speculative_eagle_topk": 1,
            "speculative_num_draft_tokens": 4,
            "attention_backend": "tokenspeed_mla",
            "decode_attention_backend": None,
            "prefill_attention_backend": None,
            "kv_cache_dtype": "fp8_e4m3",
            "page_size": 32,
        }
        values.update(overrides)
        for name, value in values.items():
            setattr(args, name, value)
        return args

    @staticmethod
    def _validate(args, *, cuda=True, hip=False):
        with (
            patch("sglang.srt.server_args.is_cuda", return_value=cuda),
            patch("sglang.srt.server_args.is_hip", return_value=hip),
        ):
            args._handle_dcp_validation()

    def test_valid_cuda_allowlist_combinations_pass(self):
        for algorithm in ("EAGLE", "EAGLE3"):
            for page_size in (32, 64):
                for comm_backend in ("a2a", "fi_a2a"):
                    with self.subTest(
                        algorithm=algorithm,
                        page_size=page_size,
                        comm_backend=comm_backend,
                    ):
                        self._validate(
                            self._make_args(
                                speculative_algorithm=algorithm,
                                page_size=page_size,
                                dcp_comm_backend=comm_backend,
                            )
                        )

    def test_tokenspeed_decode_backend_override_passes(self):
        self._validate(
            self._make_args(
                attention_backend="flashinfer",
                decode_attention_backend="tokenspeed_mla",
            )
        )

    def test_dspark_and_other_speculative_algorithms_raise(self):
        for algorithm in ("DSPARK", "DFLASH", "STANDALONE", "NGRAM"):
            with self.subTest(algorithm=algorithm):
                with self.assertRaisesRegex(ValueError, "only EAGLE or EAGLE3"):
                    self._validate(self._make_args(speculative_algorithm=algorithm))

    def test_tree_or_unresolved_topk_raises(self):
        for topk in (2, None):
            with self.subTest(topk=topk):
                with self.assertRaisesRegex(ValueError, "eagle-topk=1"):
                    self._validate(self._make_args(speculative_eagle_topk=topk))

    def test_too_many_or_unresolved_draft_tokens_raise(self):
        for draft_tokens in (5, None):
            with self.subTest(draft_tokens=draft_tokens):
                with self.assertRaisesRegex(ValueError, "draft-tokens <= 4"):
                    self._validate(
                        self._make_args(speculative_num_draft_tokens=draft_tokens)
                    )

    def test_other_target_decode_backends_raise(self):
        cases = (
            {
                "attention_backend": "flashinfer",
                "decode_attention_backend": None,
            },
            {
                "attention_backend": "flashinfer",
                "prefill_attention_backend": "tokenspeed_mla",
            },
            {
                "attention_backend": "tokenspeed_mla",
                "decode_attention_backend": "trtllm_mla",
            },
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "target decode"):
                    self._validate(self._make_args(**overrides))

    def test_other_kv_cache_dtypes_raise(self):
        for kv_cache_dtype in ("auto", "fp8_e5m2", "bfloat16"):
            with self.subTest(kv_cache_dtype=kv_cache_dtype):
                with self.assertRaisesRegex(ValueError, "fp8_e4m3"):
                    self._validate(self._make_args(kv_cache_dtype=kv_cache_dtype))

    def test_other_page_sizes_raise(self):
        for page_size in (16, 128, None):
            with self.subTest(page_size=page_size):
                with self.assertRaisesRegex(ValueError, "page-size in"):
                    self._validate(self._make_args(page_size=page_size))

    def test_draft_tokens_must_fit_in_page(self):
        with self.assertRaisesRegex(ValueError, "must be <= --page-size"):
            self._validate(
                self._make_args(
                    speculative_num_draft_tokens=4,
                    page_size=2,
                )
            )

    def test_ag_rs_comm_backend_raises(self):
        with self.assertRaisesRegex(ValueError, "a2a or fi_a2a"):
            self._validate(self._make_args(dcp_comm_backend="ag_rs"))

    def test_hip_speculative_dcp_behavior_is_unchanged(self):
        self._validate(
            self._make_args(
                dcp_comm_backend="a2a",
                speculative_algorithm="DSPARK",
                speculative_eagle_topk=8,
                speculative_num_draft_tokens=16,
                attention_backend="triton",
                kv_cache_dtype="auto",
                page_size=1,
            ),
            cuda=False,
            hip=True,
        )


if __name__ == "__main__":
    unittest.main()
