import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import create_random
from sglang.jit_kernel.moe_fused_gate import moe_fused_gate
from sglang.jit_kernel.moe_fused_gate_radix import moe_fused_gate_radix as route_radix
from sglang.jit_kernel.moe_route_radix_v2 import route_radix_v2
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=6,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
)

NUM_EXPERTS = 896
TOPK = 16
SCALE = 2.5


def _v2_sorted(scores, bias):
    return route_radix_v2(scores, bias, TOPK, True, SCALE, True, sorted=True)


def _v2_unsorted(scores, bias):
    return route_radix_v2(scores, bias, TOPK, True, SCALE, True, sorted=False)


def _v1(scores, bias):
    return route_radix(scores, bias, TOPK, True, SCALE, True)


def _triton(scores, bias):
    # fp32 scores = the triton router's production input (topk.py upcasts,
    # cost not counted here); fp32 also fails the radix covered() checks, so
    # this stays on the triton path regardless of the radix env flags.
    return moe_fused_gate(
        scores,
        bias,
        topk=TOPK,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=SCALE,
        apply_routed_scaling_factor_on_output=True,
    )


FN_MAP = {
    "triton": _triton,
    "v1": _v1,
    "v2": _v2_sorted,
    "v2_unsorted": _v2_unsorted,
}


@marker.parametrize("num_tokens", [2**n for n in range(0, 14)], [1, 64, 1024])
@marker.benchmark("provider", ["triton", "v1", "v2", "v2_unsorted"])
def benchmark(num_tokens: int, provider: str):
    torch.manual_seed(42)
    dtype = torch.float32 if provider == "triton" else torch.bfloat16
    scores = create_random(num_tokens, NUM_EXPERTS, dtype=dtype)
    bias = create_random(NUM_EXPERTS, dtype=torch.float32)
    return marker.do_bench(FN_MAP[provider], input_args=(scores, bias))


if __name__ == "__main__":
    benchmark.run()
