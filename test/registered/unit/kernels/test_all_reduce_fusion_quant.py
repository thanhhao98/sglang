"""CPU checks of the MXFP8 and NVFP4 epilogues of the push-plane boundary kernel
(``ops/communication/all_reduce_fusion.py`` over
``jit/csrc/distributed/all_reduce_fusion.cuh``): the quant ops' launch plumbing and
arch gate, the kernel's MXFP8 scale-tile arithmetic against FlashInfer's 128x4
layout, a tripwire on the place of the entry trigger (``trigger_at_entry``) in the
kernel, the NVFP4 epilogue's index arithmetic and reduction order, and two
tripwires that keep DSV4's mHC quant entry (``run_mhc_post_combine_norm_quant``)
on its own NaN rule and host checks, which the epilogue's addition left as they
were. The GPU tests are ``ops/communication/test_all_reduce_residual_norm_quant.py``
and ``ops/communication/test_all_reduce_residual_norm_quant_nvfp4.py``."""

import hashlib
import re
import unittest
from unittest.mock import Mock, patch

import numpy as np
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
                    # weight offset 0 (gemma weight pre-folded), two-shot, vectors,
                    # and the entry trigger off by default
                    self.assertEqual(args[-4:], (0.0, geometry[0], geometry[1], False))
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
        self.assertEqual(op.call_args.args[-4], 1.0)  # Gemma's weight offset
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

    def test_the_entry_trigger_reaches_the_kernel(self):
        hidden = 256
        residual = torch.zeros(8, hidden, dtype=torch.bfloat16)
        norm = (torch.ones(hidden, dtype=torch.bfloat16), 1e-6)
        for early in (False, True):
            with self.subTest(early=early):
                with patch.object(ops, "_all_reduce_residual_norm_quant_op") as op:
                    ops.all_reduce_residual_norm_quant(
                        residual,
                        residual,
                        *norm,
                        world_size=4,
                        trigger_at_entry=early,
                    )
                self.assertIs(op.call_args.args[-1], early)
                with patch.object(
                    ops, "_moe_finalize_all_reduce_residual_norm_quant_op"
                ) as op:
                    ops.moe_finalize_all_reduce_residual_norm_quant(
                        torch.zeros(8 * TOP_K, hidden, dtype=torch.bfloat16),
                        torch.zeros(8 * TOP_K, dtype=torch.int32),
                        torch.zeros(8, TOP_K),
                        TOP_K,
                        None,
                        residual,
                        *norm,
                        world_size=4,
                        hidden_dim=hidden,
                        trigger_at_entry=early,
                    )
                self.assertIs(op.call_args.args[-1], early)

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


class TestEntryTrigger(CustomTestCase):
    """trigger_at_entry: the MXFP8 form's entry trigger is the kernel's first PDL
    instruction, before its wait and before the bumper cluster branches off, so every
    CTA executes it; only the non-mHC kQuant entries take the flag, off by default;
    the end trigger after the epilogue stays."""

    def test_the_kernel_signals_at_entry_only_on_request(self):
        src = CUH.read_text()
        self.assertIn("  bool trigger_at_entry = false;\n};", src)
        body = src[src.index("void moe_finalize_all_reduce_kernel(") :]
        body = body[: body.index("// --- host")]
        entry = body.index(
            "if (params.trigger_at_entry) PDLTriggerSecondary<kUsePDL>();"
        )
        self.assertLess(entry, body.index("PDLWaitPrimary"))
        self.assertLess(entry, body.index("if (row_idx == params.num_tokens)"))
        self.assertIn(
            "if constexpr (kQuant && !kMhc) {\n"
            "    if (params.trigger_at_entry) PDLTriggerSecondary<kUsePDL>();",
            body,
        )
        # the bumper's and the epilogue's triggers are unchanged
        self.assertEqual(body.count("PDLTriggerSecondary<kUsePDL>()"), 3)
        self.assertIn("return PDLTriggerSecondary<kUsePDL>();", body)
        # the host: the two MXFP8 entries pass the flag, everything else the default
        for entry_name in ("run_residual_norm_quant", "run_plain_quant"):
            start = src.index(f"static void {entry_name}(")
            text = src[start : src.index("\n  }\n", start)]
            self.assertIn("bool trigger_at_entry) {", text)
            self.assertTrue(
                text.rstrip().endswith("std::nullopt,\n        trigger_at_entry);"),
                entry_name,
            )
        self.assertEqual(src.count("trigger_at_entry);"), 2)
        self.assertIn("bool trigger_at_entry = false) {", src)
        self.assertIn(".trigger_at_entry = trigger_at_entry,", src)


class TestQuantNvfp4Op(CustomTestCase):
    """The NVFP4 op's launch: the stock geometry for the rows, FlashInfer-sized
    outputs, the buffers it is given written in place."""

    def test_launch_geometry_and_outputs(self):
        hidden = 6144
        gsi = torch.tensor([1 / 2688])
        for rows in (1, 8, 47, 48, 64, 128):
            for ftz in (False, True):
                with self.subTest(rows=rows, ftz=ftz):
                    residual = torch.zeros(rows, hidden, dtype=torch.bfloat16)
                    with patch.object(
                        ops, "_all_reduce_residual_norm_quant_nvfp4_op"
                    ) as op:
                        out = ops.all_reduce_residual_norm_quant_nvfp4(
                            residual,
                            residual,
                            torch.ones(hidden, dtype=torch.bfloat16),
                            1e-6,
                            gsi,
                            world_size=4,
                            ftz=ftz,
                        )
                    normed, residual_out, q, s, fp4, sf, pts = out
                    self.assertIs(residual_out, residual)
                    self.assertEqual(
                        (normed.shape, normed.dtype), ((rows, hidden), torch.bfloat16)
                    )
                    self.assertEqual(q.dtype, torch.float8_e4m3fn)
                    self.assertEqual(s.shape, (ops.mxfp8_scale_bytes(hidden),))
                    self.assertEqual(
                        (fp4.shape, fp4.dtype), ((rows, hidden // 2), torch.uint8)
                    )
                    self.assertEqual(
                        (sf.shape, sf.dtype),
                        ((ops.nvfp4_scale_bytes(rows, hidden),), torch.uint8),
                    )
                    self.assertEqual((pts.shape, pts.dtype), ((rows,), torch.float32))
                    args = op.call_args.args
                    two_shot, vecs, cluster = ops._norm_launch(
                        4, rows, hidden, False, None, None, None
                    )
                    self.assertEqual(args[:4], (4, hidden, cluster, ftz))
                    self.assertEqual(args[-3:], (0.0, two_shot, vecs))
                    for i, t in enumerate((normed, q, s, fp4, sf, pts), start=4):
                        self.assertIs(args[i], t)
                    self.assertIs(args[15], gsi)

    def test_given_buffers_and_row_limits(self):
        hidden = 256
        rows = 16
        buffers = dict(
            out=torch.zeros(128, hidden, dtype=torch.bfloat16)[:rows],
            quantized=torch.zeros(128, hidden, dtype=torch.float8_e4m3fn)[:rows],
            scales=torch.zeros(ops.mxfp8_scale_bytes(hidden), dtype=torch.uint8),
            fp4=torch.zeros(128, hidden // 2, dtype=torch.uint8)[:rows],
            fp4_scales=torch.zeros(128 * hidden // 16, dtype=torch.uint8)[
                : rows * hidden // 16
            ],
            token_scales=torch.zeros(128)[:rows],
        )
        residual = torch.zeros(rows, hidden, dtype=torch.bfloat16)
        residual_out = torch.empty_like(residual)
        with patch.object(ops, "_all_reduce_residual_norm_quant_nvfp4_op") as op:
            out = ops.all_reduce_residual_norm_quant_nvfp4(
                residual,
                residual,
                torch.ones(hidden, dtype=torch.bfloat16),
                1e-6,
                torch.tensor([1.0]),
                world_size=4,
                residual_out=residual_out,
                gemma=True,
                **buffers,
            )
        for got, (name, given) in zip(
            (out[0], out[2], out[3], out[4], out[5], out[6]), buffers.items()
        ):
            self.assertEqual(got.data_ptr(), given.data_ptr(), name)
        self.assertIs(out[1], residual_out)
        self.assertEqual(op.call_args.args[-3], 1.0)  # Gemma's weight offset
        for bad in (0, 129):
            with (
                self.subTest(rows=bad),
                self.assertRaisesRegex(AssertionError, "1-128 rows"),
            ):
                big = torch.zeros(bad, hidden, dtype=torch.bfloat16)
                ops.all_reduce_residual_norm_quant_nvfp4(
                    big,
                    big,
                    torch.ones(hidden, dtype=torch.bfloat16),
                    1e-6,
                    torch.tensor([1.0]),
                    world_size=4,
                )

    def test_build_loads_the_module_each_width_launches(self):
        with patch.object(ops, "_jit_quant_nvfp4_module") as module:
            for rows, cluster in ((8, 6), (47, 6), (48, 4), (128, 4)):
                for ftz in (False, True):
                    module.reset_mock()
                    ops.load_residual_norm_quant_nvfp4(4, rows, 6144, ftz=ftz)
                    module.assert_called_once_with(4, 6144, cluster, ftz)

    def test_the_modules_template_arguments(self):
        with (
            patch.object(ops, "load_jit") as load,
            patch.object(ops, "require_cluster_launch_arch"),
            patch.object(ops, "require_nvfp4_arch"),
            patch.object(ops, "is_arch_support_pdl", return_value=True),
        ):
            ops._jit_quant_nvfp4_module.__wrapped__(4, 6144, 6, True)
            ops._jit_nvfp4_rows_module.__wrapped__(6144, 3, 2)
        (fused, rows) = load.call_args_list
        self.assertEqual(
            fused.kwargs["cuda_wrappers"],
            [
                (
                    "run",
                    "MoeFinalizeAllReduceKernel<4, 6144, 0, 6, true, bf16_t, false, "
                    "true, false, false, true, true>::run_plain_quant_nvfp4",
                )
            ],
        )
        self.assertEqual(
            rows.kwargs["cuda_wrappers"],
            [
                ("run", "Nvfp4RowQuantKernel<6144, 3, 2>::run"),
                ("run_ftz", "Nvfp4RowQuantKernel<6144, 3, 2>::run_ftz"),
            ],
        )
        self.assertNotEqual(fused.args[0], rows.args[0])

    def test_the_nvfp4_modules_need_the_kernels_arch_macro(self):
        """require_nvfp4_arch holds where the kernel's SGL_ARCH_BLACKWELL_OR_GREATER
        does, SM100 or newer and a CUDA 12.9+ nvcc (elsewhere nvfp4_quant_row fails
        the build), and both NVFP4 modules check it before compiling."""
        from sglang.kernels.jit.utils import arch

        for hip, cc, nvcc, ok in (
            (False, (10, 0), (12, 9), True),
            (False, (10, 3), (13, 0), True),
            (False, (10, 0), (12, 8), False),
            (False, (9, 0), (13, 0), False),
            (True, (10, 0), (13, 0), False),
        ):
            with (
                self.subTest(hip=hip, cc=cc, nvcc=nvcc),
                patch.object(ops, "is_hip_runtime", return_value=hip),
                patch.object(
                    ops, "get_jit_cuda_arch", return_value=arch.ArchInfo(*cc, "a")
                ),
                patch.object(arch, "_jit_cuda_version", return_value=nvcc),
            ):
                if ok:
                    ops.require_nvfp4_arch()
                else:
                    self.assertRaises(RuntimeError, ops.require_nvfp4_arch)
        with (
            patch.object(ops, "load_jit") as load,
            patch.object(ops, "require_cluster_launch_arch"),
            patch.object(ops, "require_nvfp4_arch", side_effect=RuntimeError("e2m1")),
        ):
            with self.assertRaises(RuntimeError):
                ops._jit_quant_nvfp4_module.__wrapped__(4, 6144, 6, False)
            with self.assertRaises(RuntimeError):
                ops._jit_nvfp4_rows_module.__wrapped__(6144, 6, 1)
        load.assert_not_called()

    def test_the_row_quant_entry(self):
        x = torch.zeros(5, 6144, dtype=torch.bfloat16)
        gsi = torch.tensor([1 / 2688])
        module = Mock()
        with patch.object(ops, "_jit_nvfp4_rows_module", return_value=module) as jit:
            fp4, sf, pts = ops.nvfp4_row_quant(x, gsi)
            jit.assert_called_once_with(6144, ops.default_cluster_size(6144), 1)
            module.run.assert_called_once_with(x, fp4, sf, pts, gsi)
            ops.nvfp4_row_quant(x, gsi, ftz=True, cluster_size=4, vecs_per_thread=2)
            jit.assert_called_with(6144, 4, 2)
            self.assertEqual(module.run_ftz.call_count, 1)
        self.assertEqual(
            (fp4.shape, sf.shape, pts.shape), ((5, 3072), (5 * 384,), (5,))
        )


def _vectors(cluster, vecs, hidden=6144):
    """(cluster rank, thread, v, vector) of every thread of a row's cluster, as
    all_reduce_fusion.cuh lays a row out."""
    row_vecs = hidden // 8
    block = row_vecs // (cluster * vecs)
    for rank in range(cluster):
        for tx in range(block):
            vec0 = rank * block * vecs + tx
            for v in range(vecs):
                yield rank, tx, v, vec0 + v * block


class TestEpilogueArithmetic(CustomTestCase):
    """nvfp4_quant_row's index arithmetic and reduction order, transcribed."""

    GEOMETRIES = ((6, 1), (4, 1), (3, 2), (6, 2), (4, 2), (8, 1))

    def test_every_byte_is_written_once_where_flashinfer_puts_it(self):
        hidden = 6144
        for cluster, vecs in self.GEOMETRIES:
            with self.subTest(cluster=cluster, vecs=vecs):
                codes, scales = [], []
                lanes = {}
                for rank, tx, v, vec in _vectors(cluster, vecs, hidden):
                    lanes[vec] = (rank, tx // 32, tx % 32, v)
                    codes.extend(range(vec * 4, vec * 4 + 4))
                    if vec % 2 == 0:
                        scales.append(vec // 2)
                self.assertEqual(sorted(codes), list(range(hidden // 2)))
                self.assertEqual(sorted(scales), list(range(hidden // 16)))
                for vec, (rank, warp, lane, v) in lanes.items():
                    # a block's two vectors: lanes xor 1 of one warp, the same v
                    partner = lanes[vec ^ 1]
                    self.assertEqual(partner, (rank, warp, lane ^ 1, v))
                    # FlashInfer: block b's 8 code bytes at b * 8, its first 8
                    # values (packed_lo) the low 4 bytes
                    b = vec // 2
                    self.assertEqual(vec * 4, b * 8 + 4 * (vec % 2))
                    self.assertEqual(
                        range(8 * vec, 8 * vec + 8)[0], 16 * b + 8 * (vec % 2)
                    )

    def test_the_maxima_do_not_depend_on_the_order(self):
        """max.f32 and max.bf16x2 return the other operand of a NaN (np.fmax does
        too), so FlashInfer's row amax (per 16-value block, then blocks strided over
        128 threads, warps, the CTA, from 0) equals the kernel's (per 8-value vector,
        warp butterfly, the cluster, from 0), and its block max tree equals the two
        lanes' halves joined."""
        g = np.random.default_rng(5)
        hidden = 6144
        for case in range(40):
            row = g.standard_normal(hidden).astype(np.float32) * 10.0 ** g.integers(
                -3, 4
            )
            if case % 4 == 1:
                row[g.integers(0, hidden, 50)] = np.nan
            if case % 4 == 2:
                row[g.integers(0, hidden // 16) * 16 :][:16] = np.nan
            if case % 4 == 3:
                row[:] = np.nan
                row[g.integers(0, hidden, 3)] = np.inf
            a = np.abs(row)
            # FlashInfer: per block, then each of 128 threads over its blocks, then
            # the warps' butterflies and the CTA
            blocks = [np.float32(np.nan)] * (hidden // 16)
            for b in range(hidden // 16):
                v = a[16 * b : 16 * b + 16]
                lo = _tree(v[0:8:2], v[8:16:2])
                hi = _tree(v[1:8:2], v[9:16:2])
                blocks[b] = np.fmax(lo, hi)
            local = [np.float32(0.0)] * 128
            for t in range(128):
                for b in range(t, hidden // 16, 128):
                    local[t] = np.fmax(local[t], blocks[b])
            want = np.float32(0.0)
            for t in range(128):
                want = np.fmax(want, local[t])
            # the kernel: per vector, then any order
            got = np.float32(0.0)
            for vec in g.permutation(hidden // 8):
                v = a[8 * vec : 8 * vec + 8]
                for j in range(4):
                    got = np.fmax(got, np.fmax(v[2 * j], v[2 * j + 1]))
            self.assertEqual(got, want)
            # the block max: the two lanes' halves, the first lane's first
            for b in range(hidden // 16):
                v = a[16 * b : 16 * b + 16]
                first = _tree(v[0:8:2], v[1:8:2], pairs=True)
                second = _tree(v[8:16:2], v[9:16:2], pairs=True)
                joined = (np.fmax(first[0], second[0]), np.fmax(first[1], second[1]))
                got_b = np.fmax(*joined)
                self.assertTrue(
                    (np.isnan(got_b) and np.isnan(blocks[b])) or got_b == blocks[b]
                )


def _tree(a, b, pairs=False):
    """The max.bf16x2 tree of FlashInfer's bfloat2_max_abs_8 on lanes: with
    ``pairs``, one 8-value vector's 4 words (lanes ``a``, ``b``) to one word; else the
    lane ``a`` of all 8 words (one lane's values) to a scalar."""
    if pairs:
        lo = np.fmax(np.fmax(a[0], a[1]), np.fmax(a[2], a[3]))
        hi = np.fmax(np.fmax(b[0], b[1]), np.fmax(b[2], b[3]))
        return lo, hi
    w = list(a) + list(b)
    return np.fmax(
        np.fmax(np.fmax(w[0], w[1]), np.fmax(w[2], w[3])),
        np.fmax(np.fmax(w[4], w[5]), np.fmax(w[6], w[7])),
    )


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
