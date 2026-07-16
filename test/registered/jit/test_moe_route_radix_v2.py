import pytest
import torch

from sglang.jit_kernel.moe_fused_gate_radix import moe_fused_gate_radix as route_radix
from sglang.jit_kernel.moe_route_radix_v2 import route_radix_v2
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

NUM_EXPERTS = 896
TOPK = 16


def _make_case(case: str, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.randn(
        num_tokens, NUM_EXPERTS, dtype=torch.bfloat16, device="cuda"
    )
    bias = torch.randn(NUM_EXPERTS, dtype=torch.float32, device="cuda")
    if case == "random":
        pass
    elif case == "all_equal":
        # Full 896-way key tie: min-id tie-break must pick experts 0..15.
        scores.fill_(0.5)
        bias.zero_()
    elif case == "few_values":
        # bf16-quantized duplicates: many key ties at the top-16 boundary.
        scores = (
            torch.randint(0, 8, (num_tokens, NUM_EXPERTS), device="cuda").to(
                torch.bfloat16
            )
            * 0.125
        )
        bias.zero_()
    elif case == "nan_mixed":
        scores[:, ::3] = float("nan")
    elif case == "mostly_nan":
        # Fewer than topk non-NaN entries: NaN-floored experts get selected
        # and their raw-sigmoid weights are NaN (v1 semantics).
        scores[:, : NUM_EXPERTS - 10] = float("nan")
    elif case == "huge_negative_bias":
        # biased values below the -1e30 NaN floor.
        bias = bias * 1e31 - 1e31
    else:
        raise AssertionError(case)
    return scores, bias


@pytest.mark.parametrize("num_tokens", [1, 3, 8])
@pytest.mark.parametrize(
    "case",
    [
        "random",
        "all_equal",
        "few_values",
        "nan_mixed",
        "mostly_nan",
        "huge_negative_bias",
    ],
)
@pytest.mark.parametrize(
    "renormalize,apply_scale", [(True, True), (False, False), (True, False)]
)
def test_route_radix_v2_vs_v1(num_tokens, case, renormalize, apply_scale):
    torch.manual_seed(num_tokens)
    scores, bias = _make_case(case, num_tokens)
    args = (scores, bias, TOPK, renormalize, 2.5, apply_scale)

    ref_w, ref_i = route_radix(*args)
    w, i = route_radix_v2(*args, sorted=True)
    assert torch.equal(ref_i, i), f"sorted ids diverge from v1: {case}"
    torch.testing.assert_close(ref_w, w, rtol=1e-6, atol=0.0, equal_nan=True)

    uw, ui = route_radix_v2(*args, sorted=False)
    ref_order = ref_i.argsort(dim=-1)
    u_order = ui.argsort(dim=-1)
    assert torch.equal(
        ref_i.gather(1, ref_order), ui.gather(1, u_order)
    ), f"unsorted winner set diverges from v1: {case}"
    torch.testing.assert_close(
        ref_w.gather(1, ref_order),
        uw.gather(1, u_order),
        rtol=1e-6,
        atol=0.0,
        equal_nan=True,
    )


def test_route_radix_v2_unsorted_id_order():
    # sorted=False documents compaction (expert-id ascending) output order.
    torch.manual_seed(0)
    scores = torch.randn(4, NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(NUM_EXPERTS, dtype=torch.float32, device="cuda")
    _, ids = route_radix_v2(scores, bias, TOPK, True, 2.5, True, sorted=False)
    assert torch.equal(ids, ids.sort(dim=-1).values)


def test_route_radix_v2_flag_dispatch():
    # SGLANG_OPT_USE_ROUTE_RADIX_V2=1 must route the moe_fused_gate entry
    # point to v2-unsorted: same winner set as the triton router, ids in
    # expert-id-ascending order (the v2-unsorted signature).
    from sglang.jit_kernel.moe_fused_gate import moe_fused_gate
    from sglang.srt.environ import envs

    torch.manual_seed(0)
    scores = torch.randn(2, NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(NUM_EXPERTS, dtype=torch.float32, device="cuda")
    common = dict(
        topk=TOPK,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=2.5,
        apply_routed_scaling_factor_on_output=True,
    )
    ref_w, ref_i = moe_fused_gate(scores.float(), bias, **common)  # triton
    with envs.SGLANG_OPT_USE_ROUTE_RADIX_V2.override(True):
        w, i = moe_fused_gate(scores, bias, **common)
    assert torch.equal(i, i.sort(dim=-1).values), "dispatch did not reach v2-unsorted"
    ref_order = ref_i.argsort(dim=-1)
    assert torch.equal(ref_i.to(torch.int32).gather(1, ref_order), i)
    torch.testing.assert_close(
        ref_w.gather(1, ref_order), w, rtol=1e-6, atol=0.0
    )


def test_route_radix_v2_all_equal_min_id():
    scores = torch.full(
        (2, NUM_EXPERTS), 0.5, dtype=torch.bfloat16, device="cuda"
    )
    bias = torch.zeros(NUM_EXPERTS, dtype=torch.float32, device="cuda")
    for sorted_flag in (True, False):
        _, ids = route_radix_v2(
            scores, bias, TOPK, True, 2.5, True, sorted=sorted_flag
        )
        expected = torch.arange(TOPK, dtype=torch.int32, device="cuda")
        assert torch.equal(ids, expected.expand(2, -1))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
