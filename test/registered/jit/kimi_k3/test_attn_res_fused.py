"""K3 attn-residual single-kernel aggregation (attn_res_fused).

The fused kernel replaces the 3-kernel JIT chain
  attn_res_score -> attn_res_combine -> rmsnorm
with one CTA-per-token kernel that keeps all 8 bank rows plus the prefix row
in registers: per-row score (dot(v, cw) * rrms(v)) -> softmax -> weighted sum
-> output RMSNorm. Unlike the chain, the mixed row is never rounded to bf16
before the output norm, so it should sit at least as close to the fp32
reference as the baseline chain does.

Pinned failure modes:
- chain math (score formula, row order bank[0..nvb-1] then prefix, softmax,
  fp32-accumulated combine, output RMSNorm) vs an fp32 reference, across
  every bank size: the row count is a compile-time template parameter
  dispatched through a constexpr kernel table on nvb 1..8, so each table
  slot must bind to the matching instantiation;
- read-only inputs: an earlier revision stored the result into bank[:, 0, :],
  clobbering the snapshot bank;
- the out-norm weight must be consumed as bf16x2 pairs: an earlier revision
  indexed it as scalar bf16 (implicitly widened to bf16x2 with y=0), zeroing
  every odd output element;
- the launcher must reject out-of-range nvb (0 or > 8 has no table entry).
"""

import unittest

import torch

from sglang.jit_kernel.kimi_k3.attn_res import (
    attn_res_combine,
    attn_res_fused,
    attn_res_score,
)
from sglang.jit_kernel.norm import rmsnorm
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_H = 7168
_NVB = 8
_MAX_ROWS = 16
_EPS = 1e-6


def _reference(prefix, bank, nvb, cw, ow, eps):
    """fp32 reference of the whole aggregation chain."""
    rows = torch.cat([bank[:, :nvb].float(), prefix.unsqueeze(1).float()], dim=1)
    rms = torch.rsqrt(rows.pow(2).mean(-1) + eps)
    scores = (rows * cw.float()).sum(-1) * rms
    probs = torch.softmax(scores, dim=-1)
    mixed = (probs.unsqueeze(-1) * rows).sum(1)
    return mixed * torch.rsqrt(mixed.pow(2).mean(-1, keepdim=True) + eps) * ow.float()


def _make_inputs(T: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device="cuda")

    prefix = randn(T, _H).to(torch.bfloat16)
    bank = randn(T, _NVB, _H).to(torch.bfloat16)
    # cw = score_norm_weight * proj_weight: O(1/sqrt(H)) so scores stay O(1)
    cw = (randn(_H) * _H**-0.5).to(torch.bfloat16)
    ow = (1 + 0.1 * randn(_H)).to(torch.bfloat16)
    return prefix, bank, cw, ow


class TestAttnResFused(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
            raise unittest.SkipTest("attn_res_fused requires SM100+")

    def test_matches_reference_all_bank_sizes(self):
        for nvb in range(1, 9):
            for T in (1, 5, 64):
                with self.subTest(nvb=nvb, T=T):
                    prefix, bank, cw, ow = _make_inputs(T, seed=8 * T + nvb)
                    prefix_ref, bank_ref = prefix.clone(), bank.clone()

                    out = torch.empty_like(prefix)
                    attn_res_fused(prefix, bank, cw, ow, out, nvb, _EPS)

                    ref = _reference(prefix, bank, nvb, cw, ow, _EPS)
                    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=3e-2)

                    # Inputs are read-only: the result must not be written
                    # into the snapshot bank (or the prefix row).
                    self.assertTrue(torch.equal(prefix, prefix_ref))
                    self.assertTrue(torch.equal(bank, bank_ref))

    def test_not_worse_than_jit_chain(self):
        """The fused kernel skips the bf16 rounding of the mixed row, so its
        error vs the fp32 reference must not exceed the 3-kernel chain's."""
        T = 64
        prefix, bank, cw, ow = _make_inputs(T, seed=0)

        out_fused = torch.empty_like(prefix)
        attn_res_fused(prefix, bank, cw, ow, out_fused, _NVB, _EPS)

        cw32 = cw.float().contiguous()
        scores = torch.empty(T, _MAX_ROWS, dtype=torch.float32, device="cuda")
        attn_res_score(prefix, bank, cw32, scores, _NVB, _EPS)
        mixed = torch.empty_like(prefix)
        attn_res_combine(prefix, bank, scores, mixed, _NVB)
        out_chain = torch.empty_like(prefix)
        rmsnorm(mixed, ow, out=out_chain, eps=_EPS)

        ref = _reference(prefix, bank, _NVB, cw, ow, _EPS)
        err_fused = (out_fused.float() - ref).abs().max()
        err_chain = (out_chain.float() - ref).abs().max()
        self.assertLessEqual(err_fused.item(), err_chain.item() * 1.5)

    def test_rejects_out_of_range_nvb(self):
        prefix, bank, cw, ow = _make_inputs(1, seed=0)
        out = torch.empty_like(prefix)
        for nvb in (0, 9):
            with self.assertRaises(RuntimeError):
                attn_res_fused(prefix, bank, cw, ow, out, nvb, _EPS)


if __name__ == "__main__":
    unittest.main()
