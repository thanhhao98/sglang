"""K3 attn-residual NV TMA aggregation kernel (attn_res_fused_tma).

Port of NVIDIA's warp-specialized production forward (sm_100a): 1 producer
warp streams rows into double-buffered smem via cp.async.bulk, 8 consumer
warps run per-row rms/dot reductions and an online softmax over row chunks,
with cw and ow staged in TMEM. Unlike the NV original, the output RMSNorm is
fused into the epilogue (per-thread sum of squares of the fp32 accumulator,
one cross-warp reduction, scale by rsqrt and ow).

Pinned failure modes:
- chain math (score formula, online-softmax chunk correction across 1..3
  chunks — nvb 1..8 covers full and partial chunks — fp32-accumulated
  combine, fused output RMSNorm) vs an fp32 reference, on both launcher
  configs: nvb <= 3 runs the NC=2 / 2-CTAs-per-SM config, nvb >= 4 the NC=4
  production config;
- the port's row addressing: the NV original consumed block_res
  [N-1, T, B, H]; the port reads our bank [T, NB, H] + prefix [T, H] with a
  runtime bank stride, so a bank with NB > nvb rows must still address rows
  correctly (guards the hand-adapted v_addr);
- the persistent grid and double buffering: T large enough that every CTA
  runs >= 3 chunks exercises the tb loop AND the mbarrier phase-parity flip
  (a slot is reused with flipped parity only from the third chunk on — for
  the single-chunk nvb=1 config that needs >= 3 tokens per CTA);
- read-only inputs; out-of-range nvb rejection;
- PDL under CUDA graph: a preceding kernel writes prefix_sum and the capture
  must replay bit-identically (guards capturability and the wait placement's
  basic chained correctness — the race itself is not deterministically
  testable).
"""

import unittest

import torch

from sglang.jit_kernel.kimi_k3.attn_res import attn_res_fused_tma
from sglang.srt.utils import get_device_sm
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


def _make_inputs(T: int, seed: int, num_bank_slots: int = 8):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device="cuda")

    prefix = randn(T, _H).to(torch.bfloat16)
    bank = randn(T, num_bank_slots, _H).to(torch.bfloat16)
    cw = (randn(_H) * _H**-0.5).to(torch.bfloat16)
    ow = (1 + 0.1 * randn(_H)).to(torch.bfloat16)
    return prefix, bank, cw, ow


class TestAttnResFusedTma(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
            raise unittest.SkipTest("attn_res_fused_tma requires SM100a+")

    def test_matches_reference_all_bank_sizes(self):
        # 4*SM+3: >= 3 tokens per CTA even at grid_mul=2, so every config
        # reaches gci >= 2 and flips the mbarrier phase parity.
        num_sm = torch.cuda.get_device_properties(0).multi_processor_count
        for nvb in range(1, 9):
            for T in (1, 5, 4 * num_sm + 3):
                with self.subTest(nvb=nvb, T=T):
                    prefix, bank, cw, ow = _make_inputs(T, seed=8 * T + nvb)
                    prefix_ref, bank_ref = prefix.clone(), bank.clone()

                    out = torch.empty_like(prefix)
                    attn_res_fused_tma(prefix, bank, cw, ow, out, nvb, _EPS)

                    ref = _reference(prefix, bank, nvb, cw, ow, _EPS)
                    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=4e-2)

                    self.assertTrue(torch.equal(prefix, prefix_ref))
                    self.assertTrue(torch.equal(bank, bank_ref))

    def test_wide_bank_row_addressing(self):
        """NB > nvb: rows must be addressed with the true bank stride."""
        prefix, bank, cw, ow = _make_inputs(7, seed=0, num_bank_slots=11)
        out = torch.empty_like(prefix)
        attn_res_fused_tma(prefix, bank, cw, ow, out, 5, _EPS)
        ref = _reference(prefix, bank, 5, cw, ow, _EPS)
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=4e-2)

    def test_pdl_chain_under_cuda_graph(self):
        """A preceding kernel writes prefix_sum; capture + replay (where the
        PDL launch attribute is active) must match eager execution."""
        T, nvb = 64, 8
        gen = torch.Generator(device="cuda").manual_seed(0)
        a = torch.randn(T, _H, generator=gen, device="cuda").to(torch.bfloat16)
        b = torch.randn(T, _H, generator=gen, device="cuda").to(torch.bfloat16)
        _, bank, cw, ow = _make_inputs(T, seed=1)
        prefix = torch.empty_like(a)
        out = torch.empty_like(a)

        def chain():
            torch.add(a, b, out=prefix)
            attn_res_fused_tma(prefix, bank, cw, ow, out, nvb, _EPS)

        chain()
        out_eager = out.clone()

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(3):
                chain()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                chain()
        torch.cuda.synchronize()
        out.zero_()
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out, out_eager))

    def test_rejects_out_of_range_nvb(self):
        prefix, bank, cw, ow = _make_inputs(1, seed=0)
        out = torch.empty_like(prefix)
        for nvb in (0, 9):
            with self.assertRaises(RuntimeError):
                attn_res_fused_tma(prefix, bank, cw, ow, out, nvb, _EPS)


if __name__ == "__main__":
    unittest.main()
