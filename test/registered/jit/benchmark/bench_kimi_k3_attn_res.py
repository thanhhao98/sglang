"""K3 attn-residual aggregation: fused single kernel and optimized chain vs
the baseline 3-kernel JIT chain.

Impls at K3 shapes (H=7168, bf16):
- fused:     attn_res_fused — score -> softmax -> combine -> out-RMSNorm in
             one CTA-per-token kernel; nvb dispatched via a constexpr kernel
             table; SM100+ only.
- chain:     attn_res_chain — optimized score/merge/norm kernels in one host
             call (max-width vectors + unroll for score/norm, table-dispatched
             rows and partial sum-of-squares in merge so norm needs no
             reduction); workspace allocated C++-side.
- jit_chain: baseline attn_res_score + attn_res_combine + rmsnorm.

Bandwidth is reported over the algorithmic footprint (prefix + bank rows
read, out written) for every impl, so the GB/s column is directly comparable:
extra HBM traffic (second pass over the rows, mixed-row round trip) shows up
as lower effective bandwidth.
"""

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import create_empty, create_random
from sglang.jit_kernel.kimi_k3.attn_res import (
    attn_res_chain,
    attn_res_combine,
    attn_res_fused,
    attn_res_score,
)
from sglang.jit_kernel.norm import rmsnorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=8,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
)

_H = 7168
_NB = 8
_MAX_ROWS = 16
_EPS = 1e-6


def _run_fused(prefix, bank, cw, ow, out, nvb):
    attn_res_fused(prefix, bank, cw, ow, out, nvb, _EPS)


def _run_chain(prefix, bank, cw, ow, out, nvb):
    attn_res_chain(prefix, bank, cw, ow, out, nvb, _EPS)


def _run_jit_chain(prefix, bank, cw32, ow, scores, mixed, out, nvb):
    attn_res_score(prefix, bank, cw32, scores, nvb, _EPS)
    attn_res_combine(prefix, bank, scores, mixed, nvb)
    rmsnorm(mixed, ow, out=out, eps=_EPS)


@marker.parametrize("nvb", list(range(1, 9)), [8])
@marker.parametrize("num_tokens", [2**x for x in range(14)], [1, 64])
@marker.benchmark("impl", ["fused", "chain", "jit_chain"])
def benchmark(num_tokens: int, nvb: int, impl: str):
    if impl == "fused" and torch.cuda.get_device_capability()[0] < 10:
        marker.skip("attn_res_fused requires SM100+ (fma.rn.f32.bf16)")

    prefix = create_random(num_tokens, _H)
    bank = create_random(num_tokens, _NB, _H)
    cw = (create_random(_H) * _H**-0.5).contiguous()
    ow = create_random(_H)
    out = create_empty(num_tokens, _H)

    if impl == "fused":
        args = (prefix, bank, cw, ow, out, nvb)
        fn = _run_fused
    elif impl == "chain":
        args = (prefix, bank, cw, ow, out, nvb)
        fn = _run_chain
    else:
        scores = torch.empty(
            num_tokens, _MAX_ROWS, dtype=torch.float32, device=prefix.device
        )
        mixed = create_empty(num_tokens, _H)
        args = (prefix, bank, cw.float().contiguous(), ow, scores, mixed, out, nvb)
        fn = _run_jit_chain

    return marker.do_bench(
        fn,
        input_args=args,
        graph_clone_args=(0, 1),  # prefix / bank are the read inputs
        memory_args=(prefix, bank[:, :nvb]),
        memory_output=(out,),
    )


if __name__ == "__main__":
    benchmark.run()
