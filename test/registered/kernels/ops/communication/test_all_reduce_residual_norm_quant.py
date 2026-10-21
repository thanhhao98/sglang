"""MXFP8 epilogue of SGLang's push-plane all-reduce + residual + Gemma norm boundary
kernel on 4 GPUs, in both forms: after a dense layer (plain all-reduce,
all_reduce_residual_norm_quant) and after a MoE layer (deferred finalize + shared add +
all-reduce, moe_finalize_all_reduce_residual_norm_quant), at TP4 on the
CustomAllReduceV2 push plane as MiniMax-M3.1 registers it (4 MiB push slots, 512 row
counters), at M3.1's shapes (hidden 6144, top-4, fp32 routing weights).

Against today's kernel followed by FlashInfer's mxfp8_quantize (cute-dsl, 128x4
scales, as the qkv|index GEMM quantizes its bf16 input), on the same random inputs at
every width of a decode step (1-16 and 8-128 rows): residual_out, the bf16 normed row,
the e4m3 codes and every scale byte, bit for bit, with the outputs pre-filled with
garbage, and the pair alike on every rank, with and without ``trigger_at_entry`` (the
kernel lets its PDL dependent launch at its entry); the same at forced launch
geometries (two vectors per thread).

Then in a decode step's launch order: the MXFP8 form's launch from persistent buffers
followed directly by the qkv|index GEMM on its pair (the boundary kernel is the
GEMM's PDL primary there), eagerly and captured in one CUDA graph replayed on new
inputs, against today's kernel followed by the same GEMM on its bf16 row (FlashInfer's
quantize, then the GEMM): the kernel's outputs and the GEMM's, bit for bit.

Last, rank 0 prints (nothing is checked) the MXFP8 form's time per launch against
today's kernel plus the quantize launch it replaces, from CUDA graphs. With
SGLANG_JIT_LOG_RESOURCE_USAGE=1 and a cold JIT cache, the builds also print both
kernels' register counts.

Run (one process per GPU is spawned):
    python3 -m pytest -q -s test/registered/kernels/ops/communication/test_all_reduce_residual_norm_quant.py
"""

import os
import socket
import sys
import time
import traceback
from functools import partial

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="extra-b", runner_config="4-gpu-b200")

WORLD_SIZE = 4
H = 6144
TOP_K = 4
EPS = 1e-6
N_QKV = 2560  # one TP4 rank's fused q|k|v|index_q|index_k projection
# a decode step's widths: no-spec decode steps of 1-16 rows, DSpark verify steps
# of 8-128 rows (8 a request)
ROWS = sorted(set(range(1, 17)) | set(range(8, 129, 8)))
# (rows, two_shot, vecs_per_thread, cluster_size) the default policy never picks at
# these widths: two vectors per thread
FORCED = ((8, False, 2, 3), (64, True, 2, 3), (128, True, 2, 3))
GRAPH_ROWS = (8, 64, 128)
GEMM_ROWS = (1, 8, 16, 48, 128)
# timing: where the launch policy turns two-shot (48) and to fewer clusters (64)
TIME_ROWS = (8, 16, 48, 64, 128)
TIME_LAUNCHES = 20  # launches per captured graph
TIME_REPLAYS = 10
TIMEOUT_S = 1800  # cold JIT builds of both kernel families


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= WORLD_SIZE
        and torch.cuda.get_device_capability()[0] == 10
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _init(rank: int, port: int):
    """The world group and a push-only CustomAllReduceV2 plane registered for the
    fused kernels, as test_all_reduce_residual_norm.py and M3.1 set them up."""
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


def _inputs(rows: int, rank: int, finalize: bool, seed: int):
    """One boundary's inputs. The residual stream, the norm weight and the routing are
    the same on every rank; the partial sums (the MoE's GEMM2 rows and shared expert
    output) are each rank's own. The norm weight's 32-column groups span 2^-12..2^11,
    so the UE8M0 scales span many exponents, and one 32-column group sums to zero on
    every rank (an all-zero group: scale 0)."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    mine = torch.Generator(device=dev).manual_seed(seed * 16 + rank + 1)
    residual = torch.randn(rows, H, generator=g, device=dev).bfloat16()
    spread = torch.exp2(torch.arange(H // 32, device=dev) % 24 - 12.0)
    gamma = (
        1 + 0.1 * torch.randn(H, generator=g, device=dev)
    ) * spread.repeat_interleave(32)
    gamma = gamma.bfloat16()
    zero = slice(64, 96)
    residual[:, zero] = 0
    if not finalize:
        partial = torch.randn(rows, H, generator=mine, device=dev) * (rank + 1) / 4
        partial = partial.bfloat16()
        partial[:, zero] = 0
        return dict(partial=partial, residual=residual, gamma=gamma)
    permuted = rows * TOP_K + 5
    gemm2 = (torch.randn(permuted, H, generator=mine, device=dev) / 4).bfloat16()
    gemm2[:, zero] = 0
    idx = torch.randperm(permuted, generator=g, device=dev)[: rows * TOP_K].int()
    weights = torch.rand(rows, TOP_K, generator=g, device=dev) * 2.0
    shared = (torch.randn(rows, H, generator=mine, device=dev) / 4).bfloat16()
    shared[:, zero] = 0
    return dict(
        gemm2=gemm2,
        idx=idx,
        weights=weights,
        shared=shared,
        residual=residual,
        gamma=gamma,
    )


def _stock_kernel(inp, finalize, geometry=None):
    """Today's kernel, as CustomAllReduceFusion launches it: (normed, residual_out)."""
    from sglang.kernels.ops.communication import all_reduce_fusion as ops

    launch = dict(zip(("two_shot", "vecs_per_thread", "cluster_size"), geometry or ()))
    residual_out = torch.empty_like(inp["residual"])
    if finalize:
        return ops.moe_finalize_all_reduce_residual_norm(
            inp["gemm2"],
            inp["idx"],
            inp["weights"],
            TOP_K,
            inp["shared"],
            inp["residual"],
            inp["gamma"],
            EPS,
            world_size=WORLD_SIZE,
            hidden_dim=H,
            residual_out=residual_out,
            **launch,
        )
    return ops.all_reduce_residual_norm(
        inp["partial"],
        inp["residual"],
        inp["gamma"],
        EPS,
        world_size=WORLD_SIZE,
        residual_out=residual_out,
        **launch,
    )


def _stock(inp, finalize, geometry=None):
    """Today's kernel, then FlashInfer's MXFP8 quantize of its normed row, as
    Fp8LinearMethod's GEMM path does."""
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    normed, residual_out = _stock_kernel(inp, finalize, geometry)
    q, sf = flashinfer_mxfp8_quantize(normed, is_sf_swizzled_layout=True, alignment=32)
    return normed, residual_out, q, sf


def _quant(
    inp, finalize, buffers, geometry=None, residual_out=None, trigger_at_entry=False
):
    """The MXFP8 form of the kernel, writing into ``buffers`` (normed, codes, scales);
    ``trigger_at_entry``: it signals the next kernel's launch at its entry."""
    from sglang.kernels.ops.communication import all_reduce_fusion as ops

    rows = inp["residual"].shape[0]
    normed, data, scales = buffers
    launch = dict(zip(("two_shot", "vecs_per_thread", "cluster_size"), geometry or ()))
    outputs = dict(
        residual_out=(
            torch.empty_like(inp["residual"]) if residual_out is None else residual_out
        ),
        out=normed[:rows],
        quantized=data[:rows],
        scales=scales,
        trigger_at_entry=trigger_at_entry,
        **launch,
    )
    if finalize:
        return ops.moe_finalize_all_reduce_residual_norm_quant(
            inp["gemm2"],
            inp["idx"],
            inp["weights"],
            TOP_K,
            inp["shared"],
            inp["residual"],
            inp["gamma"],
            EPS,
            world_size=WORLD_SIZE,
            hidden_dim=H,
            **outputs,
        )
    return ops.all_reduce_residual_norm_quant(
        inp["partial"],
        inp["residual"],
        inp["gamma"],
        EPS,
        world_size=WORLD_SIZE,
        **outputs,
    )


def _buffers(dev):
    """A caller's persistent buffers: sized for 128 rows, one 128-row scale tile."""
    from sglang.kernels.ops.communication import all_reduce_fusion as ops

    return (
        torch.empty(128, H, dtype=torch.bfloat16, device=dev),
        torch.empty(128, H, dtype=torch.float8_e4m3fn, device=dev),
        torch.empty(ops.mxfp8_scale_bytes(H), dtype=torch.uint8, device=dev),
    )


def _qkv_weight(dev):
    """One TP4 rank's qkv|index projection weight [N_QKV, H] in MXFP8, its scales in
    the 128x4 layout the layer keeps (weight_scale_inv_swizzled)."""
    from flashinfer import mxfp8_quantize

    g = torch.Generator(device=dev).manual_seed(4242)
    w = torch.randn(N_QKV, H, generator=g, device=dev) / H**0.5
    weight, weight_sf = mxfp8_quantize(w.bfloat16(), is_sf_swizzled_layout=True)
    return weight, weight_sf.reshape(-1)


def _gemm(x, weight, weight_sf, scales=None):
    """The qkv|index GEMM as Fp8LinearMethod runs it (cute-dsl): on the MXFP8 pair
    ``(x, scales)``, or on a bf16 row ``x`` that it quantizes first."""
    from sglang.srt.layers.quantization.fp8_utils import (
        flashinfer_mxfp8_blockscaled_linear,
    )

    return flashinfer_mxfp8_blockscaled_linear(
        x, weight, weight_sf, input_scale=scales, backend="cute-dsl"
    )


def _poison(buffers):
    """Garbage in every output byte, so a byte the kernel does not write shows."""
    normed, data, scales = buffers
    normed.fill_(float("nan"))
    data.view(torch.uint8).fill_(0x5A)
    scales.fill_(0xA5)


def _differ(name, got, want):
    """'' when bit for bit equal, else what differs."""
    if got.shape != want.shape or got.dtype != want.dtype:
        return f"{name}: {tuple(got.shape)} {got.dtype} vs {tuple(want.shape)} {want.dtype}"
    a, b = got.contiguous().view(torch.uint8), want.contiguous().view(torch.uint8)
    bad = (a != b).nonzero()
    if bad.numel() == 0:
        return ""
    return f"{name}: {bad.shape[0]} of {a.numel()} bytes differ, first at {bad[0].tolist()}"


def _compare(tag, got, want, failures):
    names = ("normed", "residual_out", "e4m3 codes", "scales")
    diffs = [d for d in (_differ(n, g, w) for n, g, w in zip(names, got, want)) if d]
    if diffs:
        failures.append(f"{tag}: " + "; ".join(diffs))
    return not diffs


def _same_on_every_rank(tag, tensors, coord, failures):
    import torch.distributed as dist

    for name, t in tensors:
        bits = t.contiguous().view(torch.uint8)
        gathered = [torch.empty_like(bits) for _ in range(coord.world_size)]
        dist.all_gather(gathered, bits, group=coord.device_group)
        for rank, other in enumerate(gathered):
            if not torch.equal(other, bits):
                failures.append(f"{tag}: {name} differs on rank {rank}")


def _check_widths(coord, rank, failures, log):
    dev = torch.device("cuda")
    buffers = _buffers(dev)
    for finalize in (False, True):
        variant = "finalize" if finalize else "plain"
        bitwise = []
        for rows in ROWS:
            inp = _inputs(rows, rank, finalize, seed=rows)
            want = _stock(inp, finalize)
            ok = True
            for early in (False, True):
                _poison(buffers)
                got = _quant(inp, finalize, buffers, trigger_at_entry=early)
                torch.cuda.synchronize()
                tag = f"{variant} rows={rows}{' entry trigger' if early else ''}"
                ok &= _compare(tag, got, want, failures)
                _same_on_every_rank(
                    tag, (("codes", got[2]), ("scales", got[3])), coord, failures
                )
            if ok:
                bitwise.append(rows)
        log(
            f"{variant}: bitwise (normed, residual_out, e4m3 codes, scales), with and "
            f"without the entry trigger, at rows {bitwise}"
        )
        for rows, *geometry in FORCED:
            inp = _inputs(rows, rank, finalize, seed=1000 + rows)
            want = _stock(inp, finalize, geometry)
            _poison(buffers)
            got = _quant(inp, finalize, buffers, geometry)
            torch.cuda.synchronize()
            tag = f"{variant} rows={rows} two_shot/vecs/cluster={geometry}"
            ok = _compare(tag, got, want, failures)
            log(f"{tag}: {'bitwise' if ok else 'DIFFERS'}")


def _check_graph_replay(coord, rank, failures, log):
    """The MXFP8 form's launch from persistent buffers and the qkv|index GEMM on its
    pair, captured in one CUDA graph as a decode step's graph holds them (the
    boundary kernel the GEMM's PDL primary) and replayed on new inputs copied into the
    captured ones; against today's kernel, FlashInfer's quantize and the same GEMM on
    the bf16 row."""
    dev = torch.device("cuda")
    buffers = _buffers(dev)
    weight, weight_sf = _qkv_weight(dev)
    for finalize in (False, True):
        variant = "finalize" if finalize else "plain"
        for rows in GRAPH_ROWS:
            static = _inputs(rows, rank, finalize, seed=2000 + rows)
            residual_out = torch.empty_like(static["residual"])

            def step():
                out = _quant(static, finalize, buffers, residual_out=residual_out)
                return out, _gemm(out[2], weight, weight_sf, scales=out[3])

            step()  # warm up: the JIT builds and the GEMM's compiled kernel
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured, captured_gemm = step()
            fresh = _inputs(rows, rank, finalize, seed=3000 + rows)
            want = _stock(fresh, finalize)
            want_gemm = _gemm(want[0], weight, weight_sf)
            for key, value in fresh.items():
                static[key].copy_(value)
            _poison(buffers)
            residual_out.fill_(float("nan"))
            captured_gemm.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            tag = f"{variant} rows={rows} graph replay"
            ok = _compare(tag, captured, want, failures)
            diff = _differ(f"{tag} qkv|index GEMM", captured_gemm, want_gemm)
            if diff:
                failures.append(diff)
            log(
                f"{tag}: boundary kernel {'bitwise' if ok else 'DIFFERS'}, "
                f"qkv|index GEMM on its pair {'bitwise' if not diff else 'DIFFERS'}"
            )


def _check_gemm(coord, rank, failures, log):
    """The qkv|index GEMM (Fp8LinearMethod's MXFP8 path) in a decode step's launch
    order: with the MXFP8 form, the boundary kernel then the GEMM on its pair, nothing
    between them; stock, today's kernel then the same GEMM on its bf16 normed row,
    which quantizes it first."""
    dev = torch.device("cuda")
    weight, weight_sf = _qkv_weight(dev)
    buffers = _buffers(dev)
    for finalize in (False, True):
        variant = "finalize" if finalize else "plain"
        for rows in GEMM_ROWS:
            # build the GEMM for these rows first, so that below the GEMM launches
            # straight after the boundary kernel
            warm = torch.zeros(rows, H, dtype=torch.bfloat16, device=dev)
            _gemm(warm, weight, weight_sf)
            inp = _inputs(rows, rank, finalize, seed=5000 + rows)
            _poison(buffers)
            torch.cuda.synchronize()
            _, _, data, scales = _quant(inp, finalize, buffers)
            got = _gemm(data, weight, weight_sf, scales=scales)
            normed, _ = _stock_kernel(inp, finalize)
            want = _gemm(normed, weight, weight_sf)
            torch.cuda.synchronize()
            diff = _differ(f"{variant} rows={rows} qkv|index GEMM", got, want)
            if diff:
                failures.append(diff)
            log(
                f"{variant} rows={rows} qkv|index GEMM on the pair: {'bitwise' if not diff else 'DIFFERS'}"
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


def _time_kernels(coord, rank, log):
    """Prints, checks nothing: the MXFP8 form's time per launch against today's kernel
    plus the quantize launch it replaces. The two share the launch geometry (bitwise
    bf16 needs it); this shows where the MXFP8 epilogue's extra work costs more than
    the quantize launch it saves."""
    dev = torch.device("cuda")
    buffers = _buffers(dev)
    for finalize in (False, True):
        variant = "finalize" if finalize else "plain"
        for rows in TIME_ROWS:
            inp = _inputs(rows, rank, finalize, seed=6000 + rows)
            residual_out = torch.empty_like(inp["residual"])
            stock = _graph_time_us(partial(_stock, inp, finalize), coord)
            quant = _graph_time_us(
                partial(_quant, inp, finalize, buffers, residual_out=residual_out),
                coord,
            )
            log(
                f"{variant} rows={rows}: today's kernel + quantize {stock:.2f} us, "
                f"MXFP8 form {quant:.2f} us ({stock - quant:+.2f} us saved per layer)"
            )


def _worker(rank: int, port: int) -> None:
    coord, comm = _init(rank, port)

    def log(message):
        if rank == 0:
            print(f"[residual_norm_quant] {message}", flush=True)

    failures = []
    try:
        with torch.inference_mode():
            _check_widths(coord, rank, failures, log)
            _check_graph_replay(coord, rank, failures, log)
            _check_gemm(coord, rank, failures, log)
            _time_kernels(coord, rank, log)
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
    not _supported(), reason=f"needs {WORLD_SIZE} SM100-family GPUs (B200/B300/GB300)"
)
def test_residual_norm_quant_matches_stock_kernel_and_flashinfer_quantize():
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
            pytest.fail(f"residual_norm_quant workers did not finish in {TIMEOUT_S} s")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
