"""Correctness test for the K3 MNNVL fused all-reduce (ar_fusion) kernels.

Compares 1shot multicast-push and in-place NVLS 2shot (with and without the
fused residual) against NCCL, bit-exact on small-int bf16 inputs, plus a
CUDA-graph capture/replay pass and a mixed stress loop.

Usage::

    python tests/test_ar_fusion.py            # relaunches under torchrun (8 GPUs)
"""

from __future__ import annotations

import atexit
import logging
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.jit_kernel.kimi_k3 import all_reduce
from sglang.jit_kernel.mp import register_comm_cleanup
from sglang.jit_kernel.tests.utils import multigpu_pytest_main
from sglang.jit_kernel.utils import cache_once, get_ci_test_range
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=240,
    stage="extra-b",
    runner_config="8-gpu-h200",
)

H = 7168  # Kimi-K3 hidden size; the kernels are tuned/used at multiples of it
MB = 1024 * 1024

PUSH_BS = [1, 2, 8, 32, 128]
PULL_BS = [1, 8, 64, 1024, 4096]
PUSH_BS = get_ci_test_range(PUSH_BS, [1, 32, 128])
PULL_BS = get_ci_test_range(PULL_BS, [1, 64, 4096])


def _precompile(num_gpus):
    for ws in num_gpus:
        all_reduce._jit_module(ws)


@cache_once
def _init_world():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    torch.cuda.set_stream(torch.cuda.Stream())
    return coord.cpu_group


@cache_once
def _init_nccl_group():
    _init_world()
    local_rank = int(os.environ["LOCAL_RANK"])
    group = dist.new_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    assert isinstance(group, dist.ProcessGroup)
    return group


@cache_once
def _init_comm() -> CustomAllReduceV2:
    cpu_group = _init_world()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    comm = CustomAllReduceV2(
        cpu_group, device, max_pull_size=1 * MB, max_push_size=2 * MB
    )
    if comm.disabled or comm.mc_base_ptr == 0:
        raise RuntimeError("ar_fusion requires CustomAllReduceV2 with multicast")
    all_reduce.register_comm(comm.obj)
    register_comm_cleanup(comm)
    return comm


@cache_once
def _init_pool_buf() -> tuple[torch.Tensor, int]:
    import torch.distributed._symmetric_memory as torch_symm_mem

    cpu_group = _init_world()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    pool = torch_symm_mem.get_mem_pool(device)
    with torch.cuda.use_mem_pool(pool):
        buf = torch.empty(max(PULL_BS) * H, dtype=torch.bfloat16, device=device)
    hdl = torch_symm_mem.rendezvous(buf, cpu_group.group_name)
    assert hdl.multicast_ptr != 0
    mc = hdl.multicast_ptr + (buf.data_ptr() - hdl.buffer_ptrs[rank])
    return buf, mc


def _device() -> torch.device:
    return torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")


def _int_input(n: int, seed: int, per_rank: bool) -> torch.Tensor:
    # small ints are exact in bf16 even after an fp32-accumulated 8-way sum,
    # so the comparison against NCCL is bit-exact
    rank = dist.get_rank() if per_rank else 0
    g = torch.Generator().manual_seed(seed * 1009 + rank)
    return torch.randint(0, 16, (n,), dtype=torch.bfloat16, generator=g).to(_device())


def _nccl_ref(x: torch.Tensor, residual):
    ref = x.clone()
    dist.all_reduce(ref, group=_init_nccl_group())
    return ref if residual is None else ref + residual


@pytest.mark.parametrize("bs", PUSH_BS)
@pytest.mark.parametrize("use_residual", [False, True])
@torch.inference_mode()
def test_ar_fusion_push(bs: int, use_residual: bool):
    comm = _init_comm()
    world = comm.world_size
    n = bs * H
    x = _int_input(n, bs, per_rank=True)
    residual = _int_input(n, bs + 7, per_rank=False) if use_residual else None
    ref = _nccl_ref(x, residual)
    all_reduce.all_reduce_push_res(world, x, residual, ws_mc_base=comm.mc_base_ptr)
    torch.cuda.synchronize()
    torch.testing.assert_close(x, ref, atol=0, rtol=0)


@pytest.mark.parametrize("bs", PULL_BS)
@pytest.mark.parametrize("use_residual", [False, True])
@torch.inference_mode()
def test_ar_fusion_pull_mc(bs: int, use_residual: bool):
    comm = _init_comm()
    world = comm.world_size
    buf, mc = _init_pool_buf()
    n = bs * H
    x = buf[:n]
    x.copy_(_int_input(n, bs + 13, per_rank=True))
    residual = _int_input(n, bs + 17, per_rank=False) if use_residual else None
    ref = _nccl_ref(x, residual)
    all_reduce.all_reduce_pull_res(world, x, residual, input_mc_ptr=mc)
    torch.cuda.synchronize()
    torch.testing.assert_close(x, ref, atol=0, rtol=0)


@torch.inference_mode()
def test_ar_fusion_stress_mixed():
    """Back-to-back mixed calls exercise the push phase double-buffering and
    the pull semaphore cycling."""
    comm = _init_comm()
    world = comm.world_size
    buf, mc = _init_pool_buf()
    for it in range(32):
        n = (1, 8, 64)[it % 3] * H
        x = _int_input(n, 3000 + it, per_rank=True)
        ref = _nccl_ref(x, None)
        all_reduce.all_reduce_push_res(world, x, None, ws_mc_base=comm.mc_base_ptr)
        torch.testing.assert_close(x, ref, atol=0, rtol=0)
        y = buf[:n]
        y.copy_(_int_input(n, 4000 + it, per_rank=True))
        ref2 = _nccl_ref(y, None)
        all_reduce.all_reduce_pull_res(world, y, None, input_mc_ptr=mc)
        torch.testing.assert_close(y, ref2, atol=0, rtol=0)


@torch.inference_mode()
def test_ar_fusion_graph_capture():
    comm = _init_comm()
    world = comm.world_size
    buf, mc = _init_pool_buf()
    cpu_group = _init_world()
    n = 64 * H
    gx = torch.zeros(n, dtype=torch.bfloat16, device=_device())
    gres = _int_input(n, 99, per_rank=False)
    gy = buf[:n]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        all_reduce.all_reduce_push_res(world, gx, gres, ws_mc_base=comm.mc_base_ptr)
        all_reduce.all_reduce_pull_res(world, gy, gres, input_mc_ptr=mc)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        all_reduce.all_reduce_push_res(world, gx, gres, ws_mc_base=comm.mc_base_ptr)
        all_reduce.all_reduce_pull_res(world, gy, gres, input_mc_ptr=mc)

    for it in range(4):
        vx = _int_input(n, 5000 + it, per_rank=True)
        vy = _int_input(n, 6000 + it, per_rank=True)
        ref_x = _nccl_ref(vx, gres)
        ref_y = _nccl_ref(vy, gres)
        gx.copy_(vx)
        gy.copy_(vy)
        dist.barrier(group=cpu_group)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(gx, ref_x, atol=0, rtol=0)
        torch.testing.assert_close(gy, ref_y, atol=0, rtol=0)


if __name__ == "__main__":
    multigpu_pytest_main(
        __name__,
        __file__,
        num_gpus=(8,),
        pre_launch_fn=_precompile,
    )
