"""Bitwise parity of the fused Step-5 DeepSelect top-512 + region pack with the two launches.

step5_select_regions_deepselect_fused runs DeepSelect's device function and the region pack
in one launch. Every output (topk_ids, region_ids, counts, packed at widths 513/514/516,
causal_lengths, and the history lengths it prepares) must equal what
step5_select_regions_deepselect writes (deep_select_topk, then the Triton _pack_regions),
bit for bit. The rows cover indexer-like scores at the 8K/32K/128K score widths, ties,
signed zeros, runs of exact zeros at the selection boundary, subnormals and infinities
(never NaN), current regions below, at and above 512, padding rows, int64 metadata,
batches of 1 to 256 rows, CUDA-graph replay, and a PDL launch behind a predecessor that
spins before it writes the scores.
"""

import pathlib
import random
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not (
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability()[0] == 10
):
    pytest.skip(
        "The Step-5 DeepSelect selection runs on NVIDIA sm_10x only.",
        allow_module_level=True,
    )

from sglang.kernels.jit.utils import load_jit  # noqa: E402
from sglang.kernels.ops.attention.step5.selection_deepselect import (  # noqa: E402
    STEP5_HISTORY_REGIONS,
    deepselect_score_width,
    step5_select_regions_deepselect,
    step5_select_regions_deepselect_fused,
)
from sglang.kernels.ops.elementwise.deep_select_topk import (  # noqa: E402
    deep_select_topk_region_pack,
)

DEVICE = "cuda"
INT32_MAX = 2**31 - 1
OUTPUTS = (
    "topk_ids",
    "region_ids",
    "counts",
    "packed",
    "causal_lengths",
    "history_lengths",
)
# About 1 ms of SM clocks: far longer than the host gap between the two launches.
SPIN_CYCLES = 2_000_000


def _indexer_like(rows, width, gen):
    # sum_h w_h * relu(q_h . k): non-negative, and exactly zero (1 in 16) when all heads are off
    weights = torch.rand(rows, 4, 1, generator=gen, device=DEVICE)
    logits = torch.randn(rows, 4, width, generator=gen, device=DEVICE)
    return (weights * logits.relu()).sum(1)


def _ties(rows, width, gen):
    return torch.randint(-3, 4, (rows, width), generator=gen, device=DEVICE).float()


def _signed_zeros(rows, width, gen):
    # +0 and -0 everywhere but a sparse positive tail: the 512th value is a zero
    u = torch.rand(rows, width, generator=gen, device=DEVICE)
    zeros = torch.where(u < 0.5, torch.zeros_like(u), -torch.zeros_like(u))
    return torch.where(u > 0.98, u, zeros)


def _zero_runs(rows, width, gen):
    # few positives, then a run of exact zeros across the selection boundary, then negatives
    u = torch.rand(rows, width, generator=gen, device=DEVICE)
    return torch.where(
        u < 0.02, 1.0 + u, torch.where(u < 0.45, torch.zeros_like(u), -1.0 - u)
    )


def _subnormals(rows, width, gen):
    bits = torch.randint(
        1, 1 << 23, (rows, width), generator=gen, device=DEVICE, dtype=torch.int32
    )
    negative = torch.rand(rows, width, generator=gen, device=DEVICE) < 0.5
    bits = torch.where(negative, bits | -(2**31), bits)
    bits[:, ::7] = 1  # repeated smallest subnormal: ties among subnormals
    return bits.view(torch.float32)


def _extremes(rows, width, gen):
    # infinities and float32 extremes over indexer-like rows; -inf stays sparse, so every
    # row keeps more than 512 values above -inf
    scores = _indexer_like(rows, width, gen)
    u = torch.rand(rows, width, generator=gen, device=DEVICE)
    finfo = torch.finfo(torch.float32)
    for low, value in (
        (0.000, float("inf")),
        (0.002, float("-inf")),
        (0.004, finfo.max),
        (0.006, -finfo.max),
        (0.008, finfo.tiny),
    ):
        scores = torch.where((u >= low) & (u < low + 0.002), value, scores)
    return scores


SCORES = dict(
    indexer_like=_indexer_like,
    ties=_ties,
    signed_zeros=_signed_zeros,
    zero_runs=_zero_runs,
    subnormals=_subnormals,
    extremes=_extremes,
)


def _mixed_currents(rows, width, seed):
    """The longest row first, then the 512 edges, then random current regions."""
    rng = random.Random(seed)
    edges = [c for c in (513, 0, 512, 1, 511, 600, 7, 100, width // 2) if c < width]
    randoms = [rng.randrange(width) for _ in range(rows)]
    return ([width - 1] + edges + randoms)[:rows]


def _make_case(
    currents,
    *,
    width,
    kind="indexer_like",
    dead=(),
    seed=0,
    dtype=torch.int32,
    slots=7,
    negative_mapping=False,
):
    """Rows whose position // 8 is `currents`; rows in `dead` use request slot 0."""
    assert max(currents) < width
    rng = random.Random(seed)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    rows = len(currents)
    positions = [8 * c + rng.randrange(8) for c in currents]
    requests = [0 if r in dead else rng.randrange(1, slots) for r in range(rows)]
    low = -(2**31) if negative_mapping else 0
    mapping = torch.randint(
        low,
        INT32_MAX,
        (slots, 8 * width + 8),
        generator=gen,
        device=DEVICE,
        dtype=torch.int32,
    )
    scores = SCORES[kind](rows, width, gen)
    assert scores.shape == (rows, width) and not scores.isnan().any()
    return SimpleNamespace(
        scores=scores,
        positions=torch.tensor(positions, dtype=dtype, device=DEVICE),
        row_to_req=torch.tensor(requests, dtype=dtype, device=DEVICE),
        req_to_token=mapping.to(dtype),
        currents=list(currents),
    )


def _outputs(rows, packed_width, fill):
    def full(shape, dtype):
        return torch.full(shape, fill, dtype=dtype, device=DEVICE)

    return dict(
        topk_ids=full((rows, STEP5_HISTORY_REGIONS), torch.int32),
        region_ids=full((rows, STEP5_HISTORY_REGIONS + 1), torch.int32),
        counts=full((rows,), torch.int32),
        packed=full((rows, packed_width), torch.int64),
        causal_lengths=full((rows,), torch.int32),
        history_lengths=full((rows,), torch.int32),
    )


def _select(op, case, out, *, history_ready=False):
    op(
        case.scores,
        case.positions,
        case.row_to_req,
        out["packed"],
        out["counts"],
        req_to_token=case.req_to_token,
        causal_lengths=out["causal_lengths"],
        history_lengths=out["history_lengths"],
        topk_ids=out["topk_ids"],
        region_ids=out["region_ids"],
        history_ready=history_ready,
    )


def _assert_same(actual, expected, names=OUTPUTS):
    for name in names:
        a, b = actual[name], expected[name]
        if not torch.equal(a, b):
            bad = (a != b).reshape(a.shape[0], -1).any(1).nonzero().flatten()
            raise AssertionError(f"{name} differs in rows {bad.tolist()[:16]}")


def _assert_fused_matches(case, *, packed_width=513, history=None):
    """Run both paths on `case` from differently pre-filled outputs; return the reference."""
    rows = case.positions.numel()
    reference = _outputs(rows, packed_width, -77)
    fused = _outputs(rows, packed_width, 0x5A5A5A5A)
    ready = history is not None
    if ready:
        reference["history_lengths"].copy_(history)
        fused["history_lengths"].copy_(history)
    _select(step5_select_regions_deepselect, case, reference, history_ready=ready)
    _select(step5_select_regions_deepselect_fused, case, fused, history_ready=ready)
    torch.cuda.synchronize()
    _assert_same(fused, reference)
    return reference


def _pack_like_triton(case, topk_ids, packed_width):
    """_pack_regions re-derived in torch from topk_ids, for non-negative metadata."""
    position = case.positions.long()
    current = position // 8
    request = case.row_to_req.long()
    live = request > 0
    count = current.clamp(max=STEP5_HISTORY_REGIONS)
    lane = torch.arange(STEP5_HISTORY_REGIONS, device=DEVICE)
    ids = topk_ids.long()
    ids = torch.where((ids >= 0) & (ids < current[:, None]), ids, INT32_MAX)
    ids = ids.sort(dim=1).values
    result = torch.where(
        lane < count[:, None],
        ids,
        torch.where(lane == count[:, None], current[:, None], -1),
    )
    full_slot = live & (count == STEP5_HISTORY_REGIONS)
    mapping = case.req_to_token.long()

    def pack(region, active):
        logical = region * 8
        slot = request[:, None].expand_as(logical)
        physical = mapping[slot, logical.clamp(0, mapping.shape[1] - 1)]
        return torch.where(active, (physical // 8) * 2**32 + logical, -1)

    packed = torch.full(
        (len(case.currents), packed_width), -1, dtype=torch.int64, device=DEVICE
    )
    packed[:, :STEP5_HISTORY_REGIONS] = pack(
        result, live[:, None] & (lane <= count[:, None])
    )
    packed[:, STEP5_HISTORY_REGIONS] = pack(current[:, None], full_slot[:, None])[:, 0]
    region_ids = torch.cat(
        [
            torch.where(live[:, None], result, -1),
            torch.where(full_slot, current, -1)[:, None],
        ],
        dim=1,
    )
    return dict(
        region_ids=region_ids.int(),
        counts=torch.where(live, count + 1, 0).int(),
        packed=packed,
        causal_lengths=torch.where(live, position + 1, 0).int(),
    )


@pytest.mark.parametrize("context", [8192, 32768, 131072])
@pytest.mark.parametrize(
    "kind", ["indexer_like", "ties", "signed_zeros", "subnormals", "extremes"]
)
def test_score_rows_at_the_step5_widths(context, kind):
    width = deepselect_score_width(context // 8)
    currents = _mixed_currents(24, width, seed=context)
    case = _make_case(currents, width=width, kind=kind, dead=(3, 17), seed=context)
    if kind == "signed_zeros":
        bits = case.scores.view(torch.int32)
        assert (bits == 0).any() and (bits == -(2**31)).any()
    _assert_fused_matches(case)


def test_runs_of_exact_zeros_at_the_selection_boundary():
    width = 4096
    rng = random.Random(5)
    currents = [rng.randrange(2000, width) for _ in range(16)]
    case = _make_case(currents, width=width, kind="zero_runs", seed=5)
    for row, current in enumerate(currents):
        assert int((case.scores[row, :current] == 0).sum()) >= 600
        assert int((case.scores[row, :current] > 0).sum()) < STEP5_HISTORY_REGIONS
    _assert_fused_matches(case)


@pytest.mark.parametrize("packed_width", [513, 514, 516])
def test_current_region_below_at_and_above_512_with_padding_rows(packed_width):
    currents = [0, 0, 1, 2, 255, 510, 511, 512, 512, 512, 513, 514, 515, 1000, 4095]
    case = _make_case(
        currents, width=4096, dead=(1, 8, 11), seed=packed_width, kind="ties"
    )
    reference = _assert_fused_matches(case, packed_width=packed_width)
    counts = reference["counts"].tolist()
    for row, current in enumerate(currents):
        assert counts[row] == (0 if row in (1, 8, 11) else min(current, 512) + 1)


@pytest.mark.parametrize("rows", [1, 4, 8, 16, 32, 64, 128, 256])
def test_mixed_batches(rows):
    width = deepselect_score_width(131072 // 8)
    currents = _mixed_currents(rows, width, seed=rows)
    dead = tuple(r for r in range(rows) if r % 5 == 3)
    case = _make_case(currents, width=width, dead=dead, seed=rows)
    _assert_fused_matches(case, packed_width=(513, 514, 516)[rows.bit_length() % 3])


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_ready_history_lengths_and_int64_metadata(dtype):
    width = 16384
    currents = _mixed_currents(32, width, seed=9)
    case = _make_case(currents, width=width, dead=(4,), seed=9, dtype=dtype)
    # Ready lengths are the caller's: shorter than the current region on long rows.
    history = torch.tensor(
        [0 if r == 4 else (c - 50 if c >= 600 else c) for r, c in enumerate(currents)],
        dtype=torch.int32,
        device=DEVICE,
    )
    _assert_fused_matches(case, packed_width=516, history=history)


def test_negative_mapping_entries_pack_with_truncating_division():
    width = 4096
    case = _make_case(
        _mixed_currents(32, width, seed=13),
        width=width,
        dead=(6,),
        seed=13,
        negative_mapping=True,
    )
    assert (case.req_to_token < 0).any()
    _assert_fused_matches(case, packed_width=514)


def test_the_reference_is_the_pack_this_test_expects():
    width = deepselect_score_width(32768 // 8)
    case = _make_case(
        _mixed_currents(40, width, seed=21), width=width, dead=(2,), seed=21
    )
    reference = _assert_fused_matches(case, packed_width=516)
    expected = _pack_like_triton(case, reference["topk_ids"], 516)
    _assert_same(reference, expected, names=tuple(expected))


def test_cuda_graph_replay_matches_eager():
    width, rows, packed_width = 16384, 32, 514
    case = _make_case(_mixed_currents(rows, width, seed=30), width=width, seed=30)
    graphed = _outputs(rows, packed_width, 0x5A5A5A5A)
    # Compile both kernels before capture.
    _select(step5_select_regions_deepselect_fused, case, graphed)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _select(step5_select_regions_deepselect_fused, case, graphed)
    for seed in (31, 32, 33):
        update = _make_case(
            _mixed_currents(rows, width, seed=seed),
            width=width,
            dead=(seed % rows,),
            seed=seed,
        )
        for name in ("scores", "positions", "row_to_req", "req_to_token"):
            getattr(case, name).copy_(getattr(update, name))
        graph.replay()
        eager = _outputs(rows, packed_width, -1)
        _select(step5_select_regions_deepselect_fused, case, eager)
        reference = _outputs(rows, packed_width, -77)
        _select(step5_select_regions_deepselect, case, reference)
        torch.cuda.synchronize()
        _assert_same(graphed, eager)
        _assert_same(graphed, reference)


def test_pdl_launch_waits_for_a_spinning_predecessor():
    spin = load_jit(
        "test_step5_pdl_spin",
        cuda_files=[str(pathlib.Path(__file__).with_name("step5_pdl_spin.cuh"))],
        cuda_wrappers=[("spin_copy", "step5_pdl_spin_copy")],
    )
    width, rows, packed_width = 16384, 64, 516
    case = _make_case(
        _mixed_currents(rows, width, seed=40), width=width, dead=(5,), seed=40
    )
    reference = _outputs(rows, packed_width, -77)
    _select(step5_select_regions_deepselect, case, reference)
    final = case.scores.clone()
    stale = _make_case(case.currents, width=width, seed=41).scores

    def launch_fused(out):
        deep_select_topk_region_pack(
            case.scores,
            out["history_lengths"],
            out["topk_ids"],
            case.positions,
            case.row_to_req,
            case.req_to_token,
            out["region_ids"],
            out["counts"],
            out["packed"],
            out["causal_lengths"],
            use_pdl=True,
        )

    # Warm-up loads both modules, so only host launch code separates the two launches.
    warm = _outputs(rows, packed_width, 0)
    warm["history_lengths"].copy_(reference["history_lengths"])
    spin.spin_copy(final.view(-1), case.scores.view(-1), 0)
    launch_fused(warm)
    torch.cuda.synchronize()
    _assert_same(warm, reference)

    fused = _outputs(rows, packed_width, 0x5A5A5A5A)
    fused["history_lengths"].copy_(reference["history_lengths"])
    case.scores.copy_(stale)
    torch.cuda.synchronize()
    # The predecessor lets the fused kernel launch at once and writes the scores ~1 ms later.
    spin.spin_copy(final.view(-1), case.scores.view(-1), SPIN_CYCLES)
    launch_fused(fused)
    torch.cuda.synchronize()
    _assert_same(fused, reference)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
