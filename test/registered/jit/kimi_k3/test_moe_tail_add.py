"""K3 MoE-tail 3-way residual add: out = bf16(bf16(a + b) + c).

The model folds the two unfused elementwise adds at the MoE tail
(up_out + shared_output, then the attn-res delayed +prefix_sum) into this
one kernel, under a bit-exactness contract: fold vs no-fold must produce
identical streams, so the kernel must round (a + b) to bf16 BEFORE adding c,
exactly like the unfused pair.

Pinned failure modes:
- double rounding: a single-rounded fp32 a + b + c "improvement" breaks the
  contract — guarded by an adversarial case where the two orders provably
  differ, plus random sweeps over the K3 shape;
- row-strided b addressing: b may be a slice of the flat concat-allreduce
  buffer (contiguous rows, row stride != H);
- covered() must stay in sync with the kernel's TensorMatcher preconditions:
  a predicate degraded to always-true crashes the model path at runtime.
"""

import unittest

import torch

from sglang.jit_kernel.kimi_k3 import moe_tail_add
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_K3_H = 7168


def _rand_bf16(*shape: int) -> torch.Tensor:
    return (torch.randn(*shape, device="cuda") * 8.0).to(torch.bfloat16)


class TestMoeTailAdd(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def _assert_bit_exact(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        self.assertTrue(moe_tail_add.covered(a, b, c))
        out = moe_tail_add.kimi_k3_moe_tail_add(a, b, c)
        ref = (a + b) + c  # the unfused pair: bf16-rounded at each step
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out, ref))

    def test_bit_exact_random(self):
        torch.manual_seed(0)
        for T in (1, 64, 128):
            for H in (_K3_H, 512):
                with self.subTest(T=T, H=H):
                    self._assert_bit_exact(
                        _rand_bf16(T, H), _rand_bf16(T, H), _rand_bf16(T, H)
                    )

    def test_double_rounding_order(self):
        # bf16 has a 7-bit mantissa, so ulp(1.0) = 2^-7 and 1 + 2^-8 is a
        # round-to-even tie back to 1.0: the unfused pair yields exactly
        # 1.0 twice, while single-rounded fp32 a + b + c gives 1 + 2^-7,
        # a different bf16. Turns red iff the kernel stops rounding
        # (a + b) before adding c.
        T, H = 2, 512
        a = torch.ones(T, H, device="cuda", dtype=torch.bfloat16)
        b = torch.full((T, H), 2.0**-8, device="cuda", dtype=torch.bfloat16)
        c = torch.full((T, H), 2.0**-8, device="cuda", dtype=torch.bfloat16)
        single_rounded = (a.float() + b.float() + c.float()).to(torch.bfloat16)
        self.assertFalse(torch.equal(single_rounded, torch.ones_like(a)))
        self._assert_bit_exact(a, b, c)

    def test_row_strided_b(self):
        # b as a slice of a wider flat buffer: rows contiguous, row stride
        # != H (the concat-allreduce layout the model hands in).
        torch.manual_seed(1)
        T, H, pad = 4, _K3_H, 256
        buf = _rand_bf16(T * (H + pad))
        b = torch.as_strided(buf, (T, H), (H + pad, 1))
        self.assertNotEqual(b.stride(0), H)
        self._assert_bit_exact(_rand_bf16(T, H), b, _rand_bf16(T, H))

    def test_covered_rejects_unsupported(self):
        # covered() gates the model-side fold; each case below violates one
        # kernel precondition and must fall back to the plain adds.
        ok = [_rand_bf16(2, 512) for _ in range(3)]
        self.assertTrue(moe_tail_add.covered(*ok))

        odd_h = [_rand_bf16(2, 500) for _ in range(3)]
        cases = {
            "fp16 dtype": (ok[0].half(), ok[1].half(), ok[2].half()),
            "empty T": tuple(t[:0] for t in ok),
            "H not multiple of 8": tuple(odd_h),
            "b column-strided": (ok[0], ok[1].t().contiguous().t(), ok[2]),
            "shape mismatch": (ok[0], ok[1][:, :256], ok[2]),
        }
        for name, (a, b, c) in cases.items():
            with self.subTest(case=name):
                self.assertFalse(moe_tail_add.covered(a, b, c))


if __name__ == "__main__":
    unittest.main()
