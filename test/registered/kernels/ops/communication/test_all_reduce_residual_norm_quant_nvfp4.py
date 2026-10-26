"""NVFP4 epilogue of SGLang's push-plane all-reduce + residual + Gemma norm boundary
kernel (all_reduce_residual_norm_quant_nvfp4: the plain all-reduce form that, besides
the MXFP8 of the normed rows, writes their per-token NVFP4), at TP4 on the
CustomAllReduceV2 push plane as MiniMax-M3.1 registers it (4 MiB push slots, 512 row
counters), at M3.1's hidden width 6144.

1 GPU, the NVFP4 epilogue on rows from memory (``nvfp4_row_quant`` at every launch
geometry, in both subnormal modes) against FlashInfer's ``nvfp4_quantize`` as SGLang's
routed MoE calls it (per-token, linear scales, cute-dsl, fast math, no 4over6): on rows
that tell the two subnormal modes and the block scale's arithmetic apart, random rows
at magnitudes 1e-30..1e30, and rows with NaN, Inf and -0. One mode must match every
byte of every row (``_flashinfer_mode`` finds it on the telling rows at the served
global scale and its f32 neighbours); the test prints which mode FlashInfer's kernel
has.

4 GPUs (one process each), the epilogue in that mode:
- at every width of a decode step (1-16 and 8-128 rows), against today's kernel
  followed by FlashInfer's MXFP8 quantize (cute-dsl, 128x4, as the gate_up quantizes
  its bf16 input) and FlashInfer's per-token NVFP4 quantize of the same bf16 rows:
  residual_out, the bf16 normed rows, the e4m3 codes, every MXFP8 scale byte, the
  packed E2M1 codes, the E4M3 block scales and the per-token scales, bit for bit, with
  the outputs pre-filled with garbage, the quantized outputs alike on every rank; the
  MXFP8-only form (all_reduce_residual_norm_quant, dense layers) likewise; both again
  at forced launch geometries (two vectors per thread);
- in a decode step's launch order: the NVFP4 form from persistent buffers followed
  directly by the shared expert's gate_up GEMM on its pair (the boundary kernel is the
  GEMM's PDL primary there), captured in one CUDA graph and replayed on new inputs,
  against today's kernel, the two quantizes and the same GEMM on the bf16 rows;
- the dense MLP's and the shared expert's gate_up GEMMs on the pair against the same
  GEMMs on the bf16 rows;
- rank 0 prints (nothing is checked) the time per launch of the NVFP4 form against
  today's kernel plus the two quantize launches it replaces, from CUDA graphs.

Run (the 4-GPU test spawns one process per GPU):
    python3 -m pytest -q -s test/registered/kernels/ops/communication/test_all_reduce_residual_norm_quant_nvfp4.py
"""

import math
import os
import socket
import sys
import time
import traceback
from functools import partial

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=420, stage="extra-b", runner_config="4-gpu-b200")

WORLD_SIZE = 4
H = 6144
SF = H // 16
EPS = 1e-6
N_SHARED = 1536  # one TP4 rank's shared-expert gate_up (2 x 3072 / 4)
N_DENSE = 6144  # one TP4 rank's dense-MLP gate_up (2 x 12288 / 4)
# a decode step's widths: no-spec decode steps of 1-16 rows, DSpark verify steps of
# 8-128 rows (8 a request)
ROWS = sorted(set(range(1, 17)) | set(range(8, 129, 8)))
# (rows, two_shot, vecs_per_thread, cluster_size) the default policy never picks at
# these widths: two vectors per thread
FORCED = ((8, False, 2, 3), (64, True, 2, 3), (128, True, 2, 3))
# the epilogue's (cluster_size, vecs_per_thread): the default geometries (one-shot
# below 48 rows, two-shot from 48) and every forced one
ROW_GEOMETRIES = ((6, 1), (4, 1), (3, 2), (6, 2), (4, 2))
GRAPH_ROWS = (8, 64, 128)
GEMM_ROWS = (1, 8, 16, 48, 128)
TIME_ROWS = (8, 16, 48, 64, 128)
TIME_LAUNCHES = 20  # launches per captured graph
TIME_REPLAYS = 10
TIMEOUT_S = (
    2400  # cold JIT builds of both kernel families and FlashInfer's CuTe kernels
)
# SGLang's routed MoE quantizes with the global scale 1 / (E4M3_MAX * 6) (448 without
# FlashInfer's 4over6)
E4M3_MAX = 448.0
# the NVFP4 outputs, in FlashInfer's order
NVFP4_OUTPUTS = ("fp4 codes", "block scales", "per-token scales")
# the inverse global scales the mode is found at, in f32 ulps from the served one
PROBE_SCALE_ULPS = (0, -1, 1, -2, 2, -3, 3)
# _midpoint_rows: a row's amax is E4M3_MAX * c, so its per-token scale is c / 6 up to
# the f32 roundings, never a power of two (c = 3/2 would make it 1/4)
MIDPOINT_ROW_C = (1.0, 17 / 16, 9 / 8, 5 / 4, 11 / 8, 13 / 8, 7 / 4, 15 / 8)


def _sm100(devices: int) -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= devices
        and torch.cuda.get_device_capability()[0] == 10
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(-1).view(torch.uint8)


def _differ(name, got, want):
    """'' when the bytes are equal, else what differs."""
    a, b = _bits(got), _bits(want)
    if a.numel() != b.numel():
        return f"{name}: {a.numel()} vs {b.numel()} bytes"
    bad = (a != b).nonzero()
    if bad.numel() == 0:
        return ""
    return f"{name}: {bad.shape[0]} of {a.numel()} bytes differ, first at byte {bad[0].item()}"


def _global_scale_inv(dev):
    """The stock chain's torch.full of 1 / (e4m3_max * 6)."""
    return torch.full((1,), 1.0 / (E4M3_MAX * 6.0), dtype=torch.float32, device=dev)


def _flashinfer_nvfp4(x, gsi):
    """FlashInfer's per-token NVFP4 quantize as SGLang's routed MoE calls it:
    ``(codes [m, H / 2], block scales [m, H / 16], per-token scales [m])``."""
    from flashinfer import SfLayout, nvfp4_quantize

    fp4, sf, pts = nvfp4_quantize(
        x,
        gsi,
        sfLayout=SfLayout.layout_linear,
        per_token_activation=True,
        backend="cute-dsl",
    )
    m = x.shape[0]
    return fp4.reshape(m, -1), sf.reshape(m, -1), pts


def _check_rows(m: int, hidden: int) -> torch.Tensor:
    """Gaussian rows of magnitudes 1e-3 to 1e3, and from row 1 on rows that reach the
    quantize's other branches: all zeros (zero amax), one value throughout, one large
    outlier, tiny values (fp32, on the CPU)."""
    g = torch.Generator(device="cpu").manual_seed(31)
    x = torch.randn(m, hidden, generator=g, dtype=torch.float32)
    x *= torch.logspace(-3, 3, m, dtype=torch.float32)[:, None]
    special = (
        torch.zeros(hidden),
        torch.full((hidden,), -1.5),
        torch.zeros(hidden).index_fill_(0, torch.tensor([hidden // 3]), 3.0e4),
        torch.randn(hidden, generator=g) * 1e-6,
    )
    for i, row in enumerate(special[: m - 1]):
        x[i + 1] = row
    return x.contiguous()


def _e4m3_values_and_midpoints(e4m3_max: float):
    """The positive finite e4m3 values up to ``e4m3_max``, and the midpoints between
    neighbouring ones (and between 0 and the smallest) below it, ascending."""
    e4m3 = torch.arange(1, 0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    mids = torch.cat((e4m3[:1] / 2, (e4m3[:-1] + e4m3[1:]) / 2))
    return e4m3[e4m3 <= e4m3_max], mids[mids < e4m3_max]


def _tie_rows(hidden: int, e4m3_max: float = E4M3_MAX) -> torch.Tensor:
    """Rows whose scales and codes land on rounding ties at a power-of-two encode
    scale: with the row's amax 6 * e4m3_max, the per-token scale at the served global
    scale is 1 (a power of two at the other magnitudes), so a block's scale is its
    max times rcp(6) alone, and a one-ulp difference in rcp(6) moves a byte: blocks
    whose max is 6x each midpoint between neighbouring e4m3 values (the block scale
    is that midpoint, up to an ulp), at three magnitudes; and blocks of max 6x an
    e4m3 value holding that value times each midpoint between neighbouring e2m1
    values. A power-of-two encode scale multiplies exactly, so at the served global
    scale the order of the multiplies cannot show here; it does on
    ``_midpoint_rows``."""
    e4m3, mids = _e4m3_values_and_midpoints(e4m3_max)
    blocks = hidden // 16
    rows = []
    for magnitude in (1.0, 2.0**-20, 2.0**10):
        x = torch.zeros(blocks, 16)
        x[0, 0] = 6.0 * e4m3_max
        for b in range(1, blocks):
            mid = mids[(b - 1) % len(mids)]
            # past the first pass, just off the ties
            x[b, 0] = 6 * mid * (1 + ((b - 1) // len(mids)) * 2.0**-7)
            x[b, 1] = -3 * mid
        rows.append(x.reshape(-1) * magnitude)
    e2m1_mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    x = torch.zeros(blocks, 16)
    x[0, 0] = 6.0 * e4m3_max
    for b in range(1, blocks):
        value = e4m3[(7 * b) % len(e4m3)]
        x[b, 0] = 6 * value
        x[b, 1:8] = value * e2m1_mids
        x[b, 8:15] = -value * e2m1_mids
    rows.append(x.reshape(-1))
    return torch.stack(rows)


def _midpoint_rows(hidden: int, e4m3_max: float = E4M3_MAX) -> torch.Tensor:
    """Rows on which the order of the block scale's multiplies shows: a row of amax
    ``e4m3_max * c`` (``MIDPOINT_ROW_C``) has blocks of max ``m * c`` for each midpoint
    ``m`` between neighbouring e4m3 values for which that is a bf16 value, so that the
    block scale ``encode * (block_max * rcp(6))`` is ``m`` in exact arithmetic at an
    encode scale near ``6 / c``, and its f32 roundings (of the per-token scale, the
    encode scale, rcp(6) and the two multiplies) decide each byte. At the global
    scales of ``_probe_scales``, which move those blocks across their midpoints by
    about an ulp a step, another association of the multiplies or a one-ulp change in
    rcp(6) changes some byte, whichever way rcp.approx rounds."""
    _, mids = _e4m3_values_and_midpoints(e4m3_max)
    mids = mids.double()
    blocks = hidden // 16
    rows = []
    for c in MIDPOINT_ROW_C:
        maxima = mids * c
        maxima = maxima[maxima.to(torch.bfloat16).double() == maxima][: blocks - 1]
        x = torch.zeros(blocks, 16, dtype=torch.float64)
        x[0, 0] = e4m3_max * c
        x[1 : 1 + len(maxima), 0] = maxima
        x[1 : 1 + len(maxima), 1] = -maxima / 2
        rows.append(x.reshape(-1))
    return torch.stack(rows).float()


def _probe_rows(hidden: int, device, e4m3_max: float = E4M3_MAX) -> torch.Tensor:
    """bf16 rows that tell the epilogue's subnormal modes and the block scale's
    arithmetic apart: ``_check_rows``' (Gaussian rows of magnitudes 1e-3 to 1e3, a zero
    row, a constant row, an outlier, tiny values); rows whose 16-value blocks span
    2^-40..2^40; rows on rounding ties at a power-of-two encode scale (``_tie_rows``)
    and at others (``_midpoint_rows``); and last, rows where the two subnormal modes
    differ: an amax near 1e-36 (a subnormal per-token scale), an amax of 1e-33 with a
    block of max 5e-38 (a subnormal block max / 6 that still makes a nonzero e4m3
    scale), and bf16 subnormals (an amax that compares equal to zero when
    flushed)."""
    g = torch.Generator(device="cpu").manual_seed(47)
    rows = [_check_rows(8, hidden)]
    blocks = hidden // 16
    spread = torch.exp2((torch.arange(blocks) % 81 - 40).float())
    rows.append(torch.randn(4, hidden, generator=g) * spread.repeat_interleave(16))
    rows.append(_tie_rows(hidden, e4m3_max))
    rows.append(_midpoint_rows(hidden, e4m3_max))
    subnormal_scale = torch.randn(hidden, generator=g) * 1e-36
    block_window = torch.randn(hidden, generator=g) * 1e-34
    block_window[7] = 1e-33
    block_window[160:176] = torch.linspace(-5e-38, 5e-38, 16)
    bf16_subnormals = torch.randn(hidden, generator=g) * 1e-39
    rows.append(torch.stack((subnormal_scale, block_window, bf16_subnormals)))
    return torch.cat(rows).to(torch.bfloat16).to(device).contiguous()


def _probe_scales(e4m3_max: float, device) -> list[tuple[int, torch.Tensor]]:
    """``(ulps, [1] fp32)`` for each of ``PROBE_SCALE_ULPS``: the inverse global scale
    SGLang's routed MoE passes (``1 / (e4m3_max * 6)``, rounded as its torch.full
    rounds it) and its f32 neighbours. Both kernels take it at run time and do the
    same arithmetic at any value; off the served one only the roundings move, so
    that the tie blocks of ``_midpoint_rows`` cross their midpoints whichever way
    rcp.approx rounds."""
    served = torch.full((1,), 1.0 / (e4m3_max * 6.0), dtype=torch.float32)
    scales = []
    for ulps in PROBE_SCALE_ULPS:
        scale = served
        toward = torch.full_like(served, math.copysign(math.inf, ulps))
        for _ in range(abs(ulps)):
            scale = torch.nextafter(scale, toward)
        scales.append((ulps, scale.to(device)))
    return scales


def _flashinfer_mode(geometries, device) -> tuple:
    """``(ftz, "")``: the subnormal mode under which the NVFP4 epilogue
    (``nvfp4_row_quant``, at each ``(cluster_size, vecs_per_thread)`` of
    ``geometries``) writes FlashInfer's ``nvfp4_quantize`` (as SGLang's routed MoE
    calls it) bit for bit on ``_probe_rows`` at every global scale of
    ``_probe_scales``; ``(None, what differs)`` when neither does."""
    from flashinfer import SfLayout, nvfp4_quantize

    from sglang.kernels.ops.communication.all_reduce_fusion import nvfp4_row_quant

    x = _probe_rows(H, device)
    differ = {False: [], True: []}
    for ulps, scale in _probe_scales(E4M3_MAX, device):
        want = nvfp4_quantize(
            x,
            scale,
            sfLayout=SfLayout.layout_linear,
            per_token_activation=True,
            backend="cute-dsl",
        )
        at = f", global scale {ulps:+d} ulps" if ulps else ""
        for ftz in (False, True):
            for cluster_size, vecs in geometries:
                got = nvfp4_row_quant(
                    x, scale, ftz=ftz, cluster_size=cluster_size, vecs_per_thread=vecs
                )
                differ[ftz] += [
                    f"{name} at cluster {cluster_size} x {vecs} vectors{at}"
                    for name, a, b in zip(NVFP4_OUTPUTS, got, want)
                    if a.numel() * a.element_size() != b.numel() * b.element_size()
                    or not torch.equal(_bits(a), _bits(b))
                ]
    for ftz in (False, True):
        if not differ[ftz]:
            return ftz, ""
    return None, (
        "the NVFP4 epilogue differs from FlashInfer's per-token quantize on the probe "
        f"rows (subnormals kept: {', '.join(differ[False][:3])}; flushed: "
        f"{', '.join(differ[True][:3])})"
    )


def _mode_name(ftz) -> str:
    return "subnormals flushed" if ftz else "subnormals kept"


# ------------------------------------------------------------ 1 GPU: the epilogue


def _special_rows(dev):
    """Rows with NaN, Inf and -0: a few NaN values, a 16-value block all NaN, an
    all-NaN row, +-Inf values, -0 throughout, an all-zero row."""
    g = torch.Generator(device="cpu").manual_seed(91)
    rows = torch.randn(6, H, generator=g)
    rows[0, torch.randint(0, H, (40,), generator=g)] = float("nan")
    rows[1, 160:176] = float("nan")
    rows[2] = float("nan")
    rows[3, torch.randint(0, H, (5,), generator=g)] = float("inf")
    rows[3, 7] = -float("inf")
    rows[4] = -0.0
    rows[5] = 0.0
    return rows.to(torch.bfloat16).to(dev).contiguous()


def _random_rows(dev, m, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(m, H, generator=g)
    x *= torch.logspace(-30, 30, m)[:, None] if m > 1 else 1.0
    return x.to(torch.bfloat16).to(dev).contiguous()


def _check_row_quant(failures, log):
    """The epilogue on rows from memory against FlashInfer, every geometry and mode;
    the mode ``_flashinfer_mode`` finds must match every row."""
    from sglang.kernels.ops.communication.all_reduce_fusion import nvfp4_row_quant

    dev = torch.device("cuda")
    gsi = _global_scale_inv(dev)
    ftz, why = _flashinfer_mode(((6, 1), (4, 1)), dev)
    if ftz is None:
        failures.append(f"no bit-for-bit mode: {why}")
    log(
        "FlashInfer's per-token NVFP4 kernel "
        + (
            f"matches the epilogue with {_mode_name(ftz)}"
            if ftz is not None
            else f"matches neither mode: {why}"
        )
    )
    sets = {
        "probe rows": _probe_rows(H, dev),
        "NaN / Inf / -0 rows": _special_rows(dev),
        **{
            f"{m} random rows": _random_rows(dev, m, seed=m)
            for m in (1, 7, 64, 128, 1000)
        },
    }
    names = ("codes", "block scales", "per-token scales")
    for label, x in sets.items():
        want = _flashinfer_nvfp4(x, gsi)
        matched = {}
        for mode in (False, True):
            for cluster, vecs in ROW_GEOMETRIES:
                got = nvfp4_row_quant(
                    x, gsi, ftz=mode, cluster_size=cluster, vecs_per_thread=vecs
                )
                diffs = [d for d in map(_differ, names, got, want) if d]
                matched.setdefault(mode, []).append(not diffs)
                if mode == ftz and diffs:
                    failures.append(
                        f"{label}, cluster {cluster} x {vecs}: " + "; ".join(diffs)
                    )
        log(
            f"{label}: bitwise with subnormals kept at "
            f"{sum(matched[False])}/{len(ROW_GEOMETRIES)} geometries, flushed at "
            f"{sum(matched[True])}/{len(ROW_GEOMETRIES)}"
        )


@pytest.mark.skipif(not _sm100(1), reason="needs an SM100-family GPU (B200/B300/GB300)")
def test_nvfp4_epilogue_matches_flashinfer_per_token_quantize():
    failures = []
    with torch.inference_mode():
        _check_row_quant(
            failures, lambda message: print(f"[nvfp4_row_quant] {message}")
        )
    assert not failures, "\n".join(failures)


# ------------------------------------------------------------ 4 GPUs: the kernel


def _init(rank: int, port: int):
    """The world group and a push-only CustomAllReduceV2 plane registered for the
    fused kernels, as test_all_reduce_residual_norm_quant.py and M3.1 set them up."""
    import torch.distributed as dist

    import sglang.srt.distributed.parallel_state as ps
    from sglang.kernels.ops.communication import all_reduce_fusion
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )
    from sglang.srt.runtime_context import get_parallel

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD_SIZE,
    )
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(WORLD_SIZE)), local_rank=rank, backend="nccl"
    )
    get_parallel().override_permanently(world_group=coord)
    comm = CustomAllReduceV2(
        coord.cpu_group,
        torch.device(f"cuda:{rank}"),
        max_pull_size=0,
        max_pull_blocks=0,
        max_push_size=4 * 1024 * 1024,
        max_push_blocks=512,
    )
    assert not comm.disabled
    all_reduce_fusion.register_comm(comm.obj)
    return coord, comm


def _inputs(rows: int, rank: int, seed: int):
    """One FFN-input boundary's inputs: the residual stream and the norm weight alike
    on every rank, the attention's partial sum each rank's own. The norm weight's
    32-column groups span 2^-12..2^11 (so do the scales of both formats), one
    32-column group is zero on every rank; from 3 rows on, row 1 is zero (a zero amax)
    and row 2 tiny (1e-30)."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    mine = torch.Generator(device=dev).manual_seed(seed * 16 + rank + 1)
    residual = torch.randn(rows, H, generator=g, device=dev)
    spread = torch.exp2(torch.arange(H // 32, device=dev) % 24 - 12.0)
    gamma = (1 + 0.1 * torch.randn(H, generator=g, device=dev)) * (
        spread.repeat_interleave(32)
    )
    partial_sum = torch.randn(rows, H, generator=mine, device=dev) * (rank + 1) / 4
    zero = slice(64, 96)
    residual[:, zero] = 0
    partial_sum[:, zero] = 0
    if rows >= 3:
        residual[1], partial_sum[1] = 0, 0
        residual[2] *= 1e-30
        partial_sum[2] *= 1e-30
    return dict(
        partial=partial_sum.bfloat16(),
        residual=residual.bfloat16(),
        gamma=gamma.bfloat16(),
    )


def _launch(geometry):
    return dict(zip(("two_shot", "vecs_per_thread", "cluster_size"), geometry or ()))


def _stock(inp, gsi, geometry=None, nvfp4=True):
    """Today's kernel (as CustomAllReduceFusion._ffn_input_add_norm launches it), then
    FlashInfer's MXFP8 quantize of its normed rows (the gate_up's) and, for a MoE
    layer, its per-token NVFP4 quantize (the routed experts')."""
    from sglang.kernels.ops.communication import all_reduce_fusion as ops
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    normed, residual_out = ops.all_reduce_residual_norm(
        inp["partial"],
        inp["residual"],
        inp["gamma"],
        EPS,
        world_size=WORLD_SIZE,
        residual_out=torch.empty_like(inp["residual"]),
        **_launch(geometry),
    )
    q, sf = flashinfer_mxfp8_quantize(normed, is_sf_swizzled_layout=True, alignment=32)
    out = (normed, residual_out, q, sf)
    return out + _flashinfer_nvfp4(normed, gsi) if nvfp4 else out


def _buffers(dev):
    """A caller's persistent buffers, sized for 128 rows."""
    from sglang.kernels.ops.communication import all_reduce_fusion as ops

    return dict(
        out=torch.empty(128, H, dtype=torch.bfloat16, device=dev),
        quantized=torch.empty(128, H, dtype=torch.float8_e4m3fn, device=dev),
        scales=torch.empty(ops.mxfp8_scale_bytes(H), dtype=torch.uint8, device=dev),
        fp4=torch.empty(128, H // 2, dtype=torch.uint8, device=dev),
        fp4_scales=torch.empty(128 * SF, dtype=torch.uint8, device=dev),
        token_scales=torch.empty(128, dtype=torch.float32, device=dev),
    )


def _poison(buffers):
    """Garbage in every output byte, so a byte the kernel does not write shows."""
    for name, t in buffers.items():
        _bits(t).fill_(0xA5 if name != "out" else 0xFF)


def _quant(inp, gsi, buffers, ftz, geometry=None, residual_out=None, nvfp4=True):
    """The quantizing kernel's launch: the NVFP4 form (MoE layers) or the MXFP8 form
    (dense layers), into ``buffers``."""
    from sglang.kernels.ops.communication import all_reduce_fusion as ops

    rows = inp["residual"].shape[0]
    outputs = dict(
        residual_out=(
            torch.empty_like(inp["residual"]) if residual_out is None else residual_out
        ),
        out=buffers["out"][:rows],
        quantized=buffers["quantized"][:rows],
        scales=buffers["scales"],
        **_launch(geometry),
    )
    args = (inp["partial"], inp["residual"], inp["gamma"], EPS)
    if not nvfp4:
        return ops.all_reduce_residual_norm_quant(
            *args, world_size=WORLD_SIZE, **outputs
        )
    out = ops.all_reduce_residual_norm_quant_nvfp4(
        *args,
        gsi,
        world_size=WORLD_SIZE,
        ftz=ftz,
        fp4=buffers["fp4"][:rows],
        fp4_scales=buffers["fp4_scales"][: rows * SF],
        token_scales=buffers["token_scales"][:rows],
        **outputs,
    )
    return out[:5] + (out[5].view(rows, SF), out[6])


NAMES = (
    "normed",
    "residual_out",
    "e4m3 codes",
    "MXFP8 scales",
    "e2m1 codes",
    "NVFP4 block scales",
    "per-token scales",
)


def _compare(tag, got, want, failures):
    diffs = [d for d in map(_differ, NAMES, got, want) if d]
    if diffs:
        failures.append(f"{tag}: " + "; ".join(diffs))
    return not diffs


def _same_on_every_rank(tag, tensors, coord, failures):
    import torch.distributed as dist

    for name, t in tensors:
        bits = _bits(t)
        gathered = [torch.empty_like(bits) for _ in range(coord.world_size)]
        dist.all_gather(gathered, bits, group=coord.device_group)
        for rank, other in enumerate(gathered):
            if not torch.equal(other, bits):
                failures.append(f"{tag}: {name} differs on rank {rank}")


def _check_widths(coord, rank, ftz, failures, log):
    dev = torch.device("cuda")
    gsi = _global_scale_inv(dev)
    buffers = _buffers(dev)
    for nvfp4 in (True, False):
        variant = "MoE (MXFP8 + NVFP4)" if nvfp4 else "dense (MXFP8)"
        bitwise = []
        for rows in ROWS:
            inp = _inputs(rows, rank, seed=rows)
            want = _stock(inp, gsi, nvfp4=nvfp4)
            _poison(buffers)
            got = _quant(inp, gsi, buffers, ftz, nvfp4=nvfp4)
            torch.cuda.synchronize()
            tag = f"{variant} rows={rows}"
            if _compare(tag, got, want, failures):
                bitwise.append(rows)
            quantized = [("e4m3 codes", got[2]), ("MXFP8 scales", got[3])]
            if nvfp4:
                quantized += [(n, t) for n, t in zip(NAMES[4:], got[4:])]
            _same_on_every_rank(tag, quantized, coord, failures)
        log(f"{variant}: bitwise (every output) at rows {bitwise}")
        for rows, *geometry in FORCED:
            inp = _inputs(rows, rank, seed=1000 + rows)
            want = _stock(inp, gsi, geometry, nvfp4=nvfp4)
            _poison(buffers)
            got = _quant(inp, gsi, buffers, ftz, geometry, nvfp4=nvfp4)
            torch.cuda.synchronize()
            tag = f"{variant} rows={rows} two_shot/vecs/cluster={geometry}"
            ok = _compare(tag, got, want, failures)
            log(f"{tag}: {'bitwise' if ok else 'DIFFERS'}")


def _weight(n, dev, seed):
    """One TP4 rank's gate_up weight [n, H] in MXFP8, its scales in the 128x4 layout
    the layer keeps (weight_scale_inv_swizzled)."""
    from flashinfer import mxfp8_quantize

    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(n, H, generator=g, device=dev) / H**0.5
    weight, weight_sf = mxfp8_quantize(w.bfloat16(), is_sf_swizzled_layout=True)
    return weight, weight_sf.reshape(-1)


def _gemm(x, weight, weight_sf, scales=None):
    """The gate_up GEMM as Fp8LinearMethod runs it (cute-dsl): on the MXFP8 pair
    ``(x, scales)``, or on bf16 rows ``x`` that it quantizes first."""
    from sglang.srt.layers.quantization.fp8_utils import (
        flashinfer_mxfp8_blockscaled_linear,
    )

    return flashinfer_mxfp8_blockscaled_linear(
        x, weight, weight_sf, input_scale=scales, backend="cute-dsl"
    )


def _check_graph_replay(coord, rank, ftz, failures, log):
    """The NVFP4 form from its persistent buffers and the shared expert's gate_up on
    its pair, captured in one CUDA graph as a decode step's graph holds them (the
    boundary kernel the GEMM's PDL primary) and replayed on new inputs copied into the
    captured ones; against today's kernel, the two quantizes and the same GEMM on the
    bf16 rows."""
    dev = torch.device("cuda")
    gsi = _global_scale_inv(dev)
    buffers = _buffers(dev)
    weight, weight_sf = _weight(N_SHARED, dev, 4242)
    for rows in GRAPH_ROWS:
        static = _inputs(rows, rank, seed=2000 + rows)
        residual_out = torch.empty_like(static["residual"])

        def step():
            out = _quant(static, gsi, buffers, ftz, residual_out=residual_out)
            return out, _gemm(out[2], weight, weight_sf, scales=out[3])

        step()  # warm up: the JIT builds and the GEMM's compiled kernel
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured, captured_gemm = step()
        fresh = _inputs(rows, rank, seed=3000 + rows)
        want = _stock(fresh, gsi)
        want_gemm = _gemm(want[0], weight, weight_sf)
        for key, value in fresh.items():
            static[key].copy_(value)
        _poison(buffers)
        residual_out.fill_(float("nan"))
        captured_gemm.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        tag = f"rows={rows} graph replay"
        ok = _compare(tag, captured, want, failures)
        diff = _differ(f"{tag} shared gate_up GEMM", captured_gemm, want_gemm)
        if diff:
            failures.append(diff)
        log(
            f"{tag}: boundary kernel {'bitwise' if ok else 'DIFFERS'}, gate_up GEMM on "
            f"its pair {'bitwise' if not diff else 'DIFFERS'}"
        )


def _check_gemm(coord, rank, ftz, failures, log):
    """The dense MLP's and the shared expert's gate_up (Fp8LinearMethod's MXFP8 path)
    in a decode step's launch order: with the quantizing kernel, the boundary kernel
    then the GEMM on its pair, nothing between them; stock, today's kernel then the
    same GEMM on its bf16 normed rows, which quantizes them first."""
    dev = torch.device("cuda")
    gsi = _global_scale_inv(dev)
    buffers = _buffers(dev)
    for name, n, nvfp4 in (
        ("shared expert", N_SHARED, True),
        ("dense MLP", N_DENSE, False),
    ):
        weight, weight_sf = _weight(n, dev, 77 + n)
        for rows in GEMM_ROWS:
            # build the GEMM for these rows first, so that below the GEMM launches
            # straight after the boundary kernel
            _gemm(
                torch.zeros(rows, H, dtype=torch.bfloat16, device=dev),
                weight,
                weight_sf,
            )
            inp = _inputs(rows, rank, seed=5000 + rows)
            _poison(buffers)
            torch.cuda.synchronize()
            out = _quant(inp, gsi, buffers, ftz, nvfp4=nvfp4)
            got = _gemm(out[2], weight, weight_sf, scales=out[3])
            want = _gemm(_stock(inp, gsi, nvfp4=False)[0], weight, weight_sf)
            torch.cuda.synchronize()
            diff = _differ(f"{name} rows={rows} gate_up GEMM", got, want)
            if diff:
                failures.append(diff)
            log(
                f"{name} rows={rows} gate_up GEMM on the pair: {'bitwise' if not diff else 'DIFFERS'}"
            )


def _graph_time_us(fn, coord) -> float:
    """Microseconds per call of ``fn``: TIME_LAUNCHES calls captured in one CUDA graph,
    replayed TIME_REPLAYS times once the ranks line up (each push-plane launch waits
    for its peers)."""
    import torch.distributed as dist

    fn()  # warm up
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(TIME_LAUNCHES):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    dist.barrier(group=coord.cpu_group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIME_REPLAYS):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / (TIME_LAUNCHES * TIME_REPLAYS)


def _time_kernels(coord, rank, ftz, log):
    """Prints, checks nothing: the NVFP4 form's time per launch against today's
    kernel plus the launches it replaces (the shared expert's MXFP8 quantize and the
    routed experts' per-token NVFP4 quantize, as SGLang's routed MoE calls it)."""
    from sglang.kernels.ops.communication import all_reduce_fusion as ops
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    dev = torch.device("cuda")
    gsi = _global_scale_inv(dev)
    buffers = _buffers(dev)
    for rows in TIME_ROWS:
        inp = _inputs(rows, rank, seed=6000 + rows)
        residual_out = torch.empty_like(inp["residual"])

        def stock():
            normed, _ = ops.all_reduce_residual_norm(
                inp["partial"],
                inp["residual"],
                inp["gamma"],
                EPS,
                world_size=WORLD_SIZE,
                residual_out=residual_out,
            )
            flashinfer_mxfp8_quantize(normed, is_sf_swizzled_layout=True, alignment=32)
            _flashinfer_nvfp4(normed, gsi)

        stock_us = _graph_time_us(stock, coord)
        nvfp4_us = _graph_time_us(
            partial(_quant, inp, gsi, buffers, ftz, residual_out=residual_out), coord
        )
        log(
            f"rows={rows}: today's kernel + two quantizes {stock_us:.2f} us, NVFP4 "
            f"form {nvfp4_us:.2f} us ({stock_us - nvfp4_us:+.2f} us saved per MoE "
            "layer)"
        )


def _worker(rank: int, port: int) -> None:
    coord, comm = _init(rank, port)

    def log(message):
        if rank == 0:
            print(f"[residual_norm_quant_nvfp4] {message}", flush=True)

    failures = []
    try:
        with torch.inference_mode():
            ftz, why = _flashinfer_mode(((6, 1), (4, 1)), torch.device("cuda"))
            if ftz is None:
                failures.append(f"no bit-for-bit mode: {why}")
                ftz = False
            log(f"epilogue mode: {_mode_name(ftz)}")
            _check_widths(coord, rank, ftz, failures, log)
            _check_graph_replay(coord, rank, ftz, failures, log)
            _check_gemm(coord, rank, ftz, failures, log)
            _time_kernels(coord, rank, ftz, log)
        torch.cuda.synchronize()
    except BaseException:
        # No collective teardown: the other ranks may be waiting in a collective
        # for this one. Exit now, so the parent stops them at once.
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    import torch.distributed as dist

    comm.close()
    dist.destroy_process_group()
    assert not failures, f"rank {rank}:\n" + "\n".join(failures)


@pytest.mark.skipif(
    not _sm100(WORLD_SIZE),
    reason=f"needs {WORLD_SIZE} SM100-family GPUs (B200/B300/GB300)",
)
def test_residual_norm_quant_nvfp4_matches_stock_kernel_and_flashinfer_quantizes():
    import torch.multiprocessing as mp

    context = mp.start_processes(
        _worker,
        args=(_free_port(),),
        nprocs=WORLD_SIZE,
        join=False,
        start_method="spawn",
    )
    deadline = time.monotonic() + TIMEOUT_S
    while not context.join(timeout=10):
        if time.monotonic() > deadline:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            pytest.fail(
                f"residual_norm_quant_nvfp4 workers did not finish in {TIMEOUT_S} s"
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
