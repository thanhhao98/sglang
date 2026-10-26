"""Benchmark the quantizing forms of the push-plane all-reduce + residual + RMSNorm
kernel (one launch) vs the stock kernel followed by FlashInfer's quantizes, at
MiniMax-M3.1's TP4 shapes (hidden 6144, top-4, fp32 routing weights, bf16 rows) on the
CustomAllReduceV2 push plane as M3.1 registers it (4 MiB push slots, 512 row counters).

Forms (``stock`` vs ``fused``):
- ``plain``    : all_reduce_residual_norm + mxfp8_quantize
                 vs all_reduce_residual_norm_quant
- ``finalize`` : moe_finalize_all_reduce_residual_norm + mxfp8_quantize
                 vs moe_finalize_all_reduce_residual_norm_quant
- ``nvfp4``    : all_reduce_residual_norm + mxfp8_quantize + per-token nvfp4_quantize
                 vs all_reduce_residual_norm_quant_nvfp4

Usage::

    python test/registered/kernels/benchmark/communication/bench_all_reduce_residual_norm_quant.py
"""

from __future__ import annotations

import atexit
import logging
import os
from functools import partial

import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import multigpu_bench_main
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication import all_reduce_fusion as ops
from sglang.srt.distributed.comm_cleanup import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.runtime_context import get_parallel, get_platform
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=300,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="requires multi-GPU, self-skips in CI",
)


# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

HIDDEN = 6144
TOP_K = 4
EPS = 1e-6
# The push-only plane M3.1 registers for these kernels.
MAX_PUSH_SIZE = 4 * 1024 * 1024
MAX_PUSH_BLOCKS = 512
# 1 / (e4m3 max * 6), the inverse global scale SGLang's routed MoE quantizes with.
NVFP4_GLOBAL_SCALE_INV = 1.0 / (448.0 * 6.0)
# Decode steps of 1-16 rows and DSpark verify steps of 8-128 (8 a request); CI keeps
# a one-shot and a two-shot launch geometry.
ROWS = [1, 2, 4, 8, 16, 32, 64, 128]
CI_ROWS = [1, 128]
FORMS = ["plain", "finalize", "nvfp4"]
PROVIDERS = ["stock", "fused"]


# ---------------------------------------------------------------------------
# Per-rank distributed init (run once per torchrun worker)
# ---------------------------------------------------------------------------


@cache_once
def _init_cpu_group() -> dist.ProcessGroup:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    get_parallel().override_permanently(world_group=coord)
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    torch.cuda.set_stream(torch.cuda.Stream())
    return coord.cpu_group


@cache_once
def _init_gpu_group() -> dist.ProcessGroup:
    _init_cpu_group()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    gpu_group = dist.new_group(backend="nccl", device_id=device)
    assert isinstance(gpu_group, dist.ProcessGroup)
    atexit.register(lambda: dist.destroy_process_group(gpu_group))
    return gpu_group


@cache_once
def _init_push_plane() -> int:
    """Register the push plane both providers stage through; returns the world size."""
    cpu_group = _init_cpu_group()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    comm = CustomAllReduceV2(
        cpu_group,
        device,
        max_pull_size=0,
        max_pull_blocks=0,
        max_push_size=MAX_PUSH_SIZE,
        max_push_blocks=MAX_PUSH_BLOCKS,
    )
    if comm.disabled:
        raise RuntimeError("JIT CustomAllReduceV2 is disabled on this system")
    register_comm_cleanup(comm)
    ops.register_comm(comm.obj)
    return comm.world_size


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _reads(form: str, rows: int) -> tuple:
    """A boundary's inputs: the residual stream, the norm weight and the routing alike
    on every rank, the partial sums (or the MoE's GEMM2 rows and shared-expert output)
    each rank's own."""
    shared = torch.Generator(device="cuda").manual_seed(rows)
    mine = torch.Generator(device="cuda").manual_seed(rows * 16 + dist.get_rank() + 1)
    residual = torch.randn(rows, HIDDEN, generator=shared, device="cuda").bfloat16()
    gamma = (1 + 0.1 * torch.randn(HIDDEN, generator=shared, device="cuda")).bfloat16()
    if form == "finalize":
        permuted = rows * TOP_K
        gemm2 = torch.randn(permuted, HIDDEN, generator=mine, device="cuda") / 4
        idx = torch.randperm(permuted, generator=shared, device="cuda").int()
        weights = torch.rand(rows, TOP_K, generator=shared, device="cuda")
        shared_out = torch.randn(rows, HIDDEN, generator=mine, device="cuda") / 4
        return gemm2.bfloat16(), idx, weights, shared_out.bfloat16(), residual, gamma
    partial_sum = torch.randn(rows, HIDDEN, generator=mine, device="cuda").bfloat16()
    if form == "plain":
        return partial_sum, residual, gamma
    global_scale_inv = torch.full(
        (1,), NVFP4_GLOBAL_SCALE_INV, dtype=torch.float32, device="cuda"
    )
    return partial_sum, residual, gamma, global_scale_inv


def _outputs(form: str, rows: int, provider: str) -> dict:
    """``residual_out`` and, for the fused forms, the caller buffers they write (as a
    decode graph keeps them)."""

    def empty(*shape: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(*shape, dtype=dtype, device="cuda")

    outputs = dict(residual_out=empty(rows, HIDDEN, dtype=torch.bfloat16))
    if provider == "stock":
        return outputs
    outputs.update(
        out=empty(rows, HIDDEN, dtype=torch.bfloat16),
        quantized=empty(rows, HIDDEN, dtype=torch.float8_e4m3fn),
        scales=empty(ops.mxfp8_scale_bytes(HIDDEN), dtype=torch.uint8),
    )
    if form == "nvfp4":
        outputs.update(
            fp4=empty(rows, HIDDEN // 2, dtype=torch.uint8),
            fp4_scales=empty(ops.nvfp4_scale_bytes(rows, HIDDEN), dtype=torch.uint8),
            token_scales=empty(rows, dtype=torch.float32),
        )
    return outputs


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


def _mxfp8_quantize(normed: torch.Tensor):
    # fp8_utils defines it on Blackwell only, so it is not imported at module import.
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    return flashinfer_mxfp8_quantize(normed, is_sf_swizzled_layout=True, alignment=32)


def _nvfp4_quantize(normed: torch.Tensor, global_scale_inv: torch.Tensor):
    from flashinfer import SfLayout, nvfp4_quantize

    return nvfp4_quantize(
        normed,
        global_scale_inv,
        sfLayout=SfLayout.layout_linear,
        per_token_activation=True,
        backend="cute-dsl",
    )


def _stock_plain(partial_sum, residual, gamma, *, world_size, residual_out):
    normed, _ = ops.all_reduce_residual_norm(
        partial_sum,
        residual,
        gamma,
        EPS,
        world_size=world_size,
        residual_out=residual_out,
    )
    return _mxfp8_quantize(normed)


def _stock_finalize(
    gemm2, idx, weights, shared_out, residual, gamma, *, world_size, residual_out
):
    normed, _ = ops.moe_finalize_all_reduce_residual_norm(
        gemm2,
        idx,
        weights,
        TOP_K,
        shared_out,
        residual,
        gamma,
        EPS,
        world_size=world_size,
        hidden_dim=HIDDEN,
        residual_out=residual_out,
    )
    return _mxfp8_quantize(normed)


def _stock_nvfp4(
    partial_sum, residual, gamma, global_scale_inv, *, world_size, residual_out
):
    normed, _ = ops.all_reduce_residual_norm(
        partial_sum,
        residual,
        gamma,
        EPS,
        world_size=world_size,
        residual_out=residual_out,
    )
    return _mxfp8_quantize(normed), _nvfp4_quantize(normed, global_scale_inv)


def _fused_plain(partial_sum, residual, gamma, *, world_size, **outputs):
    return ops.all_reduce_residual_norm_quant(
        partial_sum, residual, gamma, EPS, world_size=world_size, **outputs
    )


def _fused_finalize(
    gemm2, idx, weights, shared_out, residual, gamma, *, world_size, **outputs
):
    return ops.moe_finalize_all_reduce_residual_norm_quant(
        gemm2,
        idx,
        weights,
        TOP_K,
        shared_out,
        residual,
        gamma,
        EPS,
        world_size=world_size,
        hidden_dim=HIDDEN,
        **outputs,
    )


def _fused_nvfp4(
    partial_sum, residual, gamma, global_scale_inv, *, world_size, **outputs
):
    # ftz=False (the default); ftz only adds .ftz to one compare and three multiplies.
    return ops.all_reduce_residual_norm_quant_nvfp4(
        partial_sum,
        residual,
        gamma,
        EPS,
        global_scale_inv,
        world_size=world_size,
        **outputs,
    )


IMPLS = {
    ("plain", "stock"): _stock_plain,
    ("plain", "fused"): _fused_plain,
    ("finalize", "stock"): _stock_finalize,
    ("finalize", "fused"): _fused_finalize,
    ("nvfp4", "stock"): _stock_nvfp4,
    ("nvfp4", "fused"): _fused_nvfp4,
}


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@marker.parametrize("form", FORMS)
@marker.parametrize("rows", ROWS, CI_ROWS)
@marker.benchmark("provider", PROVIDERS)
def benchmark(form: str, rows: int, provider: str):
    if form == "nvfp4":
        try:
            ops.require_nvfp4_arch()
        except RuntimeError as error:
            marker.skip(str(error))
    world_size = _init_push_plane()
    gpu_group = _init_gpu_group()
    return marker.do_bench(
        partial(IMPLS[form, provider], world_size=world_size),
        input_args=_reads(form, rows),
        input_kwargs=_outputs(form, rows, provider),
        # Every call reads its own copy of the inputs; the outputs are write targets.
        graph_clone_kwargs=None,
        sync_multigpu_fn=lambda: dist.barrier(gpu_group),
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    if not get_platform().is_sm100:
        print("[skip] needs SM100-family GPUs (B200, B300, GB300)")
    else:
        multigpu_bench_main(
            name=__name__,
            file=__file__,
            num_gpus=[4],  # M3.1's TP4
            main_fn=benchmark.run,
        )
