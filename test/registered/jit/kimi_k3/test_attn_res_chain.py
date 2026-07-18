"""K3 attn-residual optimized 3-kernel chain (attn_res_chain).

One host call launches score -> merge -> norm, tuned for K3 shapes (H=7168):
max-width vectorization + unroll 2 for score and norm, a row loop dispatched
through a compile-time kernel table in merge, and per-chunk partial
sum-of-squares handed from merge to norm so the output RMSNorm needs no
reduction. The scores / partials / mixed workspace is allocated C++-side in
one allocation and never crosses the FFI boundary.

Pinned failure modes:
- chain math vs an fp32 reference across every supported bank size (the merge
  row loop is dispatched on nvb 1..8 — each table slot must bind to the
  matching instantiation: an off-by-one reads the wrong number of rows);
- workspace slicing: scores, partials and mixed are carved out of a single
  allocation — a wrong slice offset makes the kernels stomp on each other's
  buffers, which the reference comparison catches;
- the norm scale must equal rms of the mixed row: merge's per-chunk partials
  and norm's in-thread sum of exactly the kNumSplit valid slots must compose
  to the full-H sum of squares — a wrong chunk index, a missed chunk, or
  reading a pad slot shifts the whole output scale;
- read-only inputs (prefix / bank);
- the launcher rejects out-of-range nvb (0 or > 8 has no table entry).
"""

import unittest

import torch

from sglang.jit_kernel.kimi_k3.attn_res import attn_res_chain
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_H = 7168
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
    bank = randn(T, 8, _H).to(torch.bfloat16)
    cw = (randn(_H) * _H**-0.5).to(torch.bfloat16)
    ow = (1 + 0.1 * randn(_H)).to(torch.bfloat16)
    return prefix, bank, cw, ow


class TestAttnResChain(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def test_matches_reference_all_bank_sizes(self):
        for nvb in range(1, 9):
            for T in (1, 5, 64):
                with self.subTest(nvb=nvb, T=T):
                    prefix, bank, cw, ow = _make_inputs(T, seed=8 * T + nvb)
                    prefix_ref, bank_ref = prefix.clone(), bank.clone()

                    out = torch.empty_like(prefix)
                    attn_res_chain(prefix, bank, cw, ow, out, nvb, _EPS)

                    ref = _reference(prefix, bank, nvb, cw, ow, _EPS)
                    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=4e-2)

                    self.assertTrue(torch.equal(prefix, prefix_ref))
                    self.assertTrue(torch.equal(bank, bank_ref))

    def test_rejects_out_of_range_nvb(self):
        prefix, bank, cw, ow = _make_inputs(1, seed=0)
        out = torch.empty_like(prefix)
        for nvb in (0, 9):
            with self.assertRaises(RuntimeError):
                attn_res_chain(prefix, bank, cw, ow, out, nvb, _EPS)


if __name__ == "__main__":
    unittest.main()
