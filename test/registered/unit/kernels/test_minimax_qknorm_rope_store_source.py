"""CPU tripwires on the fused MiniMax-M3.1 q/k/index norm + RoPE + FP8 q + NVFP4
K/V/index-K store (``jit/csrc/minimax/fused_qknorm_rope_store.cuh``): it runs the
q/k norm kernel's code first, its PDL wait inside, and reads the store's inputs only
after it, so a predecessor that lets it launch early (a GEMM that signals its
dependent before its own wait) cannot hand it rows it has not written; and it reuses
both stock kernels' device code rather than rewriting the math. The GPU test is
``ops/attention/test_minimax_qknorm_rope_store.py``."""

import unittest

from sglang.kernels.jit.utils import KERNEL_PATH
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

CUH = KERNEL_PATH / "csrc" / "minimax" / "fused_qknorm_rope_store.cuh"


class TestFusedKernelSource(CustomTestCase):
    """The fused kernel runs the q/k norm kernel's code first (its PDL wait inside)
    and reads the store's inputs only after it. It reuses both stock kernels' device
    code rather than rewriting the math."""

    def test_order(self):
        src = CUH.read_text()
        self.assertIn('#include "fused_gemma_qknorm_rope.cuh"', src)
        self.assertIn('#include "nvfp4_store_kv_index.cuh"', src)
        self.assertIn("minimax_nvfp4::quantize_block(", src)
        self.assertNotIn("PDLWaitPrimary", src)
        self.assertNotIn("PDLTriggerSecondary", src)
        forward = src[src.index("SGL_DEVICE static void forward(") :]
        forward = forward[: forward.index("\n  }\n")]
        norm = forward.index("Norm::forward(params.norm);")
        self.assertNotIn("params.", forward[:norm].split("{", 1)[1])
        sync = forward.index("__syncwarp();")
        self.assertLess(norm, sync)
        self.assertLess(sync, forward.index("store_block<T>("))
        # the K / index-K rows this kernel wrote are read back with coherent loads
        store = src[src.index("store_block(const Nvfp4StoreParams& params") :]
        self.assertIn("volatile", store[: store.index("quantize_block(")])


if __name__ == "__main__":
    unittest.main()
