"""CPU checks of the MXFP8 epilogue of the push-plane boundary kernel
(``ops/communication/all_reduce_fusion.py`` over
``jit/csrc/distributed/all_reduce_fusion.cuh``): the quant ops' launch plumbing,
the kernel's scale-tile arithmetic against FlashInfer's 128x4 layout, and two
tripwires that keep DSV4's mHC quant entry (``run_mhc_post_combine_norm_quant``)
on its own NaN rule and host checks, which the epilogue's addition left as they
were. The GPU test is ``ops/communication/test_all_reduce_residual_norm_quant.py``."""

import hashlib
import re
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.kernels.jit.utils import KERNEL_PATH
from sglang.kernels.ops.communication import all_reduce_fusion as ops
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

TOP_K = 4
CUH = KERNEL_PATH / "csrc" / "distributed" / "all_reduce_fusion.cuh"
# sha256 of the mHC entry's code before the MXFP8 epilogue was added: mhc_quant_vec,
# its template line through its closing brace, and the body of run_impl's kQuant
# checks (now its kQuant && kMhc block)
MHC_QUANT_VEC_SHA256 = (
    "4264dfb9bdccfafcc5a012f4381cc0cff023ca8865b720b82375f146150990ed"
)
MHC_HOST_CHECKS_SHA256 = (
    "2ddcd8e50901cba575d1d4e170de2471bc9a6eff3389dc0bab02a4efca913ff4"
)


class TestQuantOps(CustomTestCase):
    """The quant ops' launch: the stock geometry for the rows, FlashInfer-sized
    outputs, and the buffers they are given written in place."""

    def test_launch_geometry_and_outputs(self):
        hidden = 6144
        for finalize in (False, True):
            for rows in (1, 8, 47, 48, 64, 128):
                with self.subTest(finalize=finalize, rows=rows):
                    op = Mock()
                    gemm2 = torch.zeros(rows * TOP_K, hidden, dtype=torch.bfloat16)
                    residual = torch.zeros(rows, hidden, dtype=torch.bfloat16)
                    name = (
                        "_moe_finalize_all_reduce_residual_norm_quant_op"
                        if finalize
                        else "_all_reduce_residual_norm_quant_op"
                    )
                    with patch.object(ops, name, op):
                        if finalize:
                            out = ops.moe_finalize_all_reduce_residual_norm_quant(
                                gemm2,
                                torch.zeros(rows * TOP_K, dtype=torch.int32),
                                torch.zeros(rows, TOP_K),
                                TOP_K,
                                None,
                                residual,
                                torch.ones(hidden, dtype=torch.bfloat16),
                                1e-6,
                                world_size=4,
                                hidden_dim=hidden,
                            )
                        else:
                            out = ops.all_reduce_residual_norm_quant(
                                residual,
                                residual,
                                torch.ones(hidden, dtype=torch.bfloat16),
                                1e-6,
                                world_size=4,
                            )
                    normed, residual_out, quantized, scales = out
                    self.assertIs(residual_out, residual)
                    self.assertEqual(normed.shape, (rows, hidden))
                    self.assertEqual(normed.dtype, torch.bfloat16)
                    self.assertEqual(quantized.shape, (rows, hidden))
                    self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
                    # FlashInfer's 128x4 scale size for up to 128 rows
                    self.assertEqual(scales.shape, (128 * hidden // 32,))
                    self.assertEqual(scales.dtype, torch.uint8)
                    args = op.call_args.args
                    geometry = ops._norm_launch(
                        4, rows, hidden, finalize, None, None, None
                    )
                    # weight offset 0 (gemma weight pre-folded), two-shot, vectors
                    self.assertEqual(args[-3:], (0.0, geometry[0], geometry[1]))
                    self.assertEqual(args[3 if finalize else 2], geometry[2])
                    self.assertIs(args[4 if finalize else 3], normed)
                    self.assertIs(args[5 if finalize else 4], quantized)
                    self.assertIs(args[6 if finalize else 5], scales)

    def test_given_buffers_and_row_limits(self):
        hidden = 256
        data = torch.zeros(128, hidden, dtype=torch.float8_e4m3fn)
        scales = torch.zeros(ops.mxfp8_scale_bytes(hidden), dtype=torch.uint8)
        normed = torch.zeros(128, hidden, dtype=torch.bfloat16)
        residual = torch.zeros(16, hidden, dtype=torch.bfloat16)
        residual_out = torch.empty_like(residual)
        with patch.object(ops, "_all_reduce_residual_norm_quant_op") as op:
            out = ops.all_reduce_residual_norm_quant(
                residual,
                residual,
                torch.ones(hidden, dtype=torch.bfloat16),
                1e-6,
                world_size=4,
                residual_out=residual_out,
                out=normed[:16],
                quantized=data[:16],
                scales=scales,
                gemma=True,
            )
        self.assertEqual(out[0].data_ptr(), normed.data_ptr())
        self.assertIs(out[1], residual_out)
        self.assertEqual(out[2].data_ptr(), data.data_ptr())
        self.assertIs(out[3], scales)
        self.assertEqual(op.call_args.args[-3], 1.0)  # Gemma's weight offset
        for rows in (0, 129):
            with (
                self.subTest(rows=rows),
                self.assertRaisesRegex(AssertionError, "1-128 rows"),
            ):
                big = torch.zeros(rows, hidden, dtype=torch.bfloat16)
                ops.all_reduce_residual_norm_quant(
                    big,
                    big,
                    torch.ones(hidden, dtype=torch.bfloat16),
                    1e-6,
                    world_size=4,
                )

    def test_build_loads_the_module_each_width_launches(self):
        with patch.object(ops, "_jit_quant_module") as module:
            for rows, cluster in ((8, 6), (47, 6), (48, 4), (128, 4)):
                module.reset_mock()
                ops.load_residual_norm_quant(4, rows, 6144)
                module.assert_called_once_with(4, 6144, 0, cluster, torch.bfloat16)
                module.reset_mock()
                ops.load_residual_norm_quant(
                    4, rows, 6144, top_k=4, weight_dtype=torch.float32
                )
                module.assert_called_once_with(4, 6144, 4, cluster, torch.float32)


def _flashinfer_sf_offset(row, col, padded_cols):
    """FlashInfer's compute_sf_index_swizzled_128x4_gpu."""
    return (
        col % 4
        + (col // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + (row // 128) * 128 * padded_cols
    )


class TestScaleLayoutArithmetic(CustomTestCase):
    """all_reduce_fusion.cuh's scale-tile index arithmetic, transcribed: the epilogue's
    offset is FlashInfer's, and the bumper's padding loop visits every byte of the
    tile once, decoding the row each byte belongs to, at one and two vectors per
    thread."""

    def test_epilogue_offsets_are_flashinfers(self):
        hidden = 6144
        groups = hidden // 32
        seen = set()
        for token in range(128):
            # the lanes holding each 32-column group's first vector write its scale
            for hvec in range(0, hidden // 8, 4):
                g = hvec // 4
                off = (
                    (g // 4) * 512 + ((token % 32) * 4 + (token // 32) % 4) * 4 + g % 4
                )
                self.assertEqual(off, _flashinfer_sf_offset(token, g, groups))
                seen.add(off)
        self.assertEqual(seen, set(range(ops.mxfp8_scale_bytes(hidden))))

    def test_bumper_zeros_exactly_the_padding_rows(self):
        hidden = 6144
        size = ops.mxfp8_scale_bytes(hidden)
        row_vecs = hidden // 8
        for cluster, vecs in ((6, 1), (4, 1), (3, 2), (6, 2)):
            block = row_vecs // (cluster * vecs)
            visited = []
            for cluster_rank in range(cluster):
                for tx in range(block):
                    off = cluster_rank * block + tx
                    while off < size:
                        visited.append(off)
                        off += cluster * block
            with self.subTest(cluster=cluster, vecs=vecs):
                self.assertEqual(sorted(visited), list(range(size)))
            for off in range(size):
                swizzled_row = (off % 512) // 4
                row = swizzled_row // 4 + (swizzled_row % 4) * 32
                col = (off // 512) * 4 + off % 4
                self.assertEqual(_flashinfer_sf_offset(row, col, hidden // 32), off)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _device_function(text: str, name: str) -> str:
    """A device function's text, its template line through its closing brace."""
    found = re.search(
        rf"^template <[^\n]*>\nSGL_DEVICE void {name}\(.*?^\}}\n", text, re.M | re.S
    )
    assert found, f"{name} not found"
    return found.group(0)


def _run_impl_blocks(text: str, condition: str) -> list[str]:
    """The bodies of run_impl's top-level ``if constexpr (<condition>)`` blocks."""
    run_impl = text[text.index("  static void run_impl(") :]
    return re.findall(
        rf"^    if constexpr \({re.escape(condition)}\) \{{\n(.*?)^    \}}\n",
        run_impl,
        re.M | re.S,
    )


class TestMhcQuantPathIsUnchanged(CustomTestCase):
    """The MXFP8 epilogue's emitter (mxfp8_quant_vec) takes FlashInfer's NaN rule,
    and its host checks are stricter; DSV4's mHC entry, which shares the epilogue,
    keeps its own emitter and checks, byte for byte."""

    def test_mhc_quant_rule_is_unchanged(self):
        text = CUH.read_text()
        self.assertEqual(
            _sha256(_device_function(text, "mhc_quant_vec")), MHC_QUANT_VEC_SHA256
        )
        # the norm epilogue's kMhc instances call it, with its original arguments, and
        # nothing else does
        dispatch = re.search(
            r"if constexpr \(kQuant\) \{\s*if constexpr \(kMhc\) \{\s*"
            r"(\w+)<kHiddenDim>\(params, out, row_idx, hvec\);\s*\} else \{\s*"
            r"(\w+)<kHiddenDim>\(params, out, row_idx, hvec \+ v \* kBlockSize\);",
            text,
        )
        self.assertIsNotNone(dispatch)
        self.assertEqual(dispatch.groups(), ("mhc_quant_vec", "mxfp8_quant_vec"))
        self.assertEqual(text.count("mhc_quant_vec<"), 1)

    def test_mhc_host_checks_are_unchanged(self):
        text = CUH.read_text()
        (mhc,) = _run_impl_blocks(text, "kQuant && kMhc")
        self.assertEqual(_sha256(mhc), MHC_HOST_CHECKS_SHA256)
        # no block applies to both: the epilogue's checks are the non-mHC entries' only
        self.assertEqual(_run_impl_blocks(text, "kQuant"), [])
        checks = _run_impl_blocks(text, "kQuant && !kMhc")
        self.assertEqual(len(checks), 2)
        for check in (
            "quantized.has_value() && scales.has_value()",
            ".with_strides({kHiddenDim, 1})",
            ".with_strides({1})",
            "num_tokens == 128 || num_tokens < push.num_blocks",
        ):
            with self.subTest(check=check):
                self.assertIn(check, "".join(checks))
                self.assertNotIn(check, mhc)


if __name__ == "__main__":
    unittest.main()
