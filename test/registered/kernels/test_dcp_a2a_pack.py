"""Bit-exactness tests for the fused DCP a2a send-buffer pack.

`dcp_a2a_pack_triton` replaces the permute + contiguous + two strided copies that
`dcp_a2a_lse_reduce` used to build its [N, B, H_local, D + LPD] payload, and
`dcp_a2a_lse_view` replaces the staging tensor + copy that unpacked the received
LSE. Both are pure data movement, so the bar is BIT-EXACT equality with the
unfused construction -- not a tolerance. A wrong port must fail here rather than
show up as subtly wrong logits in a perf run.

Covers:
1. Packed payload is byte-identical to the unfused build (dtypes x shapes x N)
2. The fp32 LSE view aliases exactly the bytes the unfused path unstaged
3. Non-contiguous / strided inputs (the caller does not guarantee contiguity)
4. End-to-end: dcp_lse_combine_triton over the fused buffers == over unfused ones
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")


def _unfused_pack(cp_attn_out, cp_attn_lse, N):
    """The exact construction dcp_a2a_lse_reduce used before the fusion."""
    from sglang.kernels.ops.attention.dcp_kernels import _lse_pack_dim

    B, H, D = cp_attn_out.shape
    H_per_rank = H // N
    out_dtype = cp_attn_out.dtype
    lpd = _lse_pack_dim(out_dtype)

    reshaped_out = cp_attn_out.view(B, N, H_per_rank, D).permute(1, 0, 2, 3)
    reshaped_lse = cp_attn_lse.view(B, N, H_per_rank).permute(1, 0, 2)

    send_lse_contig = reshaped_lse.contiguous()
    send_combined = torch.empty(
        N, B, H_per_rank, D + lpd, dtype=out_dtype, device=cp_attn_out.device
    )
    send_combined[:, :, :, :D].copy_(reshaped_out)
    send_combined[:, :, :, D:].copy_(
        send_lse_contig.view(out_dtype).view(N, B, H_per_rank, lpd)
    )
    return send_combined


def _unfused_unstage_lse(recv_combined, N, B, H_per_rank, D):
    from sglang.kernels.ops.attention.dcp_kernels import _lse_pack_dim

    out_dtype = recv_combined.dtype
    lpd = _lse_pack_dim(out_dtype)
    stg = torch.empty(
        N, B, H_per_rank, dtype=torch.float32, device=recv_combined.device
    )
    stg.view(out_dtype).view(N, B, H_per_rank, lpd).copy_(recv_combined[:, :, :, D:])
    return stg


class TestDCPA2APackBitExact(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required for Triton kernel tests")
        cls.device = "cuda"

    def _mk(self, B, H, D, dtype, seed=0):
        g = torch.Generator(device=self.device).manual_seed(seed)
        out = torch.randn(B, H, D, generator=g, device=self.device, dtype=torch.float32)
        lse = torch.randn(B, H, generator=g, device=self.device, dtype=torch.float32)
        return out.to(dtype), lse

    def _assert_pack_bit_exact(self, B, H, D, N, dtype):
        from sglang.kernels.ops.attention.dcp_kernels import (
            dcp_a2a_lse_view,
            dcp_a2a_pack_triton,
        )

        out, lse = self._mk(B, H, D, dtype)
        ref = _unfused_pack(out, lse, N)
        got, got_lse_view = dcp_a2a_pack_triton(out, lse, N)

        self.assertEqual(ref.shape, got.shape)
        # Byte-level equality: these are copies, so anything short of identical
        # bytes means the layout or the fp32 LSE alias is wrong.
        self.assertTrue(
            torch.equal(ref.view(torch.uint8), got.view(torch.uint8)),
            f"packed payload differs (B={B} H={H} D={D} N={N} {dtype})",
        )

        # The returned view must alias the same fp32 values the unfused path
        # would have unstaged from the tail columns.
        H_per_rank = H // N
        ref_lse = _unfused_unstage_lse(ref, N, B, H_per_rank, D)
        self.assertTrue(
            torch.equal(ref_lse, got_lse_view),
            f"LSE view differs (B={B} H={H} D={D} N={N} {dtype})",
        )
        # ...and it must be the ORIGINAL lse, permuted -- catches a self-consistent
        # but wrong layout that both paths would share.
        self.assertTrue(
            torch.equal(
                lse.view(B, N, H_per_rank).permute(1, 0, 2).contiguous(), got_lse_view
            ),
            "LSE view does not match the permuted source LSE",
        )
        # dcp_a2a_lse_view over the same buffer must agree with the returned view.
        self.assertTrue(torch.equal(dcp_a2a_lse_view(got), got_lse_view))

    def test_bit_exact_across_shapes(self):
        # D=512 is the MLA kv_lora_rank this path actually runs at.
        for dtype in (torch.bfloat16, torch.float16):
            for N in (2, 4, 8):
                for B, H, D in ((1, 128, 512), (7, 64, 512), (16, 128, 512)):
                    if H % N:
                        continue
                    with self.subTest(dtype=dtype, N=N, B=B, H=H, D=D):
                        self._assert_pack_bit_exact(B, H, D, N, dtype)

    def test_bit_exact_small_head_dim(self):
        for N in (2, 4):
            with self.subTest(N=N):
                self._assert_pack_bit_exact(4, 32, 64, N, torch.bfloat16)

    def test_non_contiguous_input(self):
        """The kernel reads through strides; the caller does not promise contiguity."""
        from sglang.kernels.ops.attention.dcp_kernels import dcp_a2a_pack_triton

        B, H, D, N = 4, 64, 512, 8
        dtype = torch.bfloat16
        g = torch.Generator(device=self.device).manual_seed(3)
        # Produce a genuinely strided view by slicing a wider tensor.
        wide = torch.randn(
            B, H, D * 2, generator=g, device=self.device, dtype=torch.float32
        ).to(dtype)
        out = wide[:, :, ::2]
        lse = torch.randn(B, H, generator=g, device=self.device, dtype=torch.float32)
        self.assertFalse(out.is_contiguous())

        ref = _unfused_pack(out, lse, N)
        got, _ = dcp_a2a_pack_triton(out, lse, N)
        self.assertTrue(torch.equal(ref.view(torch.uint8), got.view(torch.uint8)))

    def test_combine_end_to_end_matches(self):
        """Loopback (rank sees its own payload): fused vs unfused into the combine."""
        from sglang.kernels.ops.attention.dcp_kernels import (
            dcp_a2a_lse_view,
            dcp_a2a_pack_triton,
            dcp_lse_combine_triton,
        )

        B, H, D, N = 4, 64, 512, 8
        H_per_rank = H // N
        dtype = torch.bfloat16
        out, lse = self._mk(B, H, D, dtype, seed=11)

        ref_buf = _unfused_pack(out, lse, N)
        ref_lse = _unfused_unstage_lse(ref_buf, N, B, H_per_rank, D)
        ref_o, _ = dcp_lse_combine_triton(
            ref_buf[:, :, :, :D], ref_lse, is_lse_base_on_e=True
        )

        got_buf, _ = dcp_a2a_pack_triton(out, lse, N)
        got_o, _ = dcp_lse_combine_triton(
            got_buf[:, :, :, :D], dcp_a2a_lse_view(got_buf), is_lse_base_on_e=True
        )
        self.assertTrue(torch.equal(ref_o, got_o), "combined output differs")


if __name__ == "__main__":
    unittest.main()
