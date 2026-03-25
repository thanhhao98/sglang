"""
Benchmark DCP communication backends: All-to-All (A2A) vs AllGather+ReduceScatter (AG+RS).

Simulates the actual communication patterns used in Decode Context Parallelism (DCP)
for MLA attention. Benchmarks both eager and CUDA-graph-captured paths.

A2A path (dcp_a2a.py):
  - Fuses output + LSE into [N, B, H/N, D+2] and does a single all_to_all_single.

AG+RS path (utils.py cp_lse_ag_out_rs):
  - all_gather on LSE [B, H] -> [N*B, H]
  - reduce_scatter on corrected output [B, N*H_local, D] -> [B, H_local, D]

Usage:
    torchrun --nproc_per_node=8 benchmark_dcp_comm.py
    torchrun --nproc_per_node=8 benchmark_dcp_comm.py --batch-sizes 1,4,16,64,256
"""

import argparse
import os
from typing import List

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    SymmetricMemoryContext,
)
from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_group,
    graph_capture,
    initialize_model_parallel,
)

IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark DCP A2A vs AG+RS communication"
    )
    parser.add_argument(
        "--num-heads", type=int, default=16,
        help="Total number of attention heads (H). Default: 16 (DeepSeek-V2-Lite MLA)",
    )
    parser.add_argument(
        "--head-dim", type=int, default=512,
        help="Head dimension (D). Default: 512 (MLA absorbed head dim)",
    )
    parser.add_argument(
        "--batch-sizes", type=str, default="1,4,16,64,256",
        help="Comma-separated batch sizes to benchmark",
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Warmup iterations",
    )
    parser.add_argument(
        "--test-loop", type=int, default=100, help="Timing iterations",
    )
    parser.add_argument(
        "--graph-loop", type=int, default=100,
        help="Ops per CUDA graph replay (for graph benchmarks)",
    )
    parser.add_argument(
        "--skip-eager", action="store_true", help="Skip eager benchmarks",
    )
    parser.add_argument(
        "--skip-graph", action="store_true", help="Skip CUDA graph benchmarks",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# A2A communication (mirrors dcp_a2a.py fused path)
# ---------------------------------------------------------------------------


def a2a_fused_comm(
    send_combined: torch.Tensor,
    recv_combined: torch.Tensor,
    group: ProcessGroup,
):
    """Single all_to_all_single on fused output+LSE buffer."""
    dist.all_to_all_single(recv_combined.view(-1), send_combined.view(-1), group=group)


def a2a_fused_comm_pynccl(
    send_combined: torch.Tensor,
    recv_combined: torch.Tensor,
    pynccl_comm: PyNcclCommunicator,
):
    pynccl_comm.all_to_all_single(recv_combined.view(-1), send_combined.view(-1))


# ---------------------------------------------------------------------------
# AG+RS communication (mirrors cp_lse_ag_out_rs in utils.py)
# ---------------------------------------------------------------------------


def ag_rs_comm(
    attn_out: torch.Tensor,
    attn_lse: torch.Tensor,
    ag_lse_output: torch.Tensor,
    rs_output: torch.Tensor,
    group: ProcessGroup,
):
    """AllGather on LSE + ReduceScatter on output."""
    dist.all_gather_into_tensor(ag_lse_output, attn_lse, group=group)
    dist.reduce_scatter_tensor(rs_output, attn_out, group=group)


def ag_rs_comm_pynccl(
    attn_out: torch.Tensor,
    attn_lse: torch.Tensor,
    ag_lse_output: torch.Tensor,
    rs_output: torch.Tensor,
    pynccl_comm: PyNcclCommunicator,
):
    pynccl_comm.all_gather(ag_lse_output, attn_lse)
    pynccl_comm.reduce_scatter(rs_output, attn_out)


# ---------------------------------------------------------------------------
# Benchmarking helpers
# ---------------------------------------------------------------------------


def bench_eager(func, warmup: int, test_loop: int) -> float:
    """Benchmark an eager function, returns average time in microseconds."""
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(test_loop):
        func()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / test_loop * 1000  # us


def bench_cuda_graph(
    func,
    warmup: int,
    graph_loop: int,
    test_loop: int,
) -> float:
    """Benchmark inside a CUDA graph, returns average time in microseconds."""
    with graph_capture() as graph_capture_context:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=graph_capture_context.stream):
            for _ in range(graph_loop):
                func()

    graph.replay()

    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    latencies: List[float] = []
    for _ in range(test_loop):
        torch.cuda.synchronize()
        dist.barrier()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        latencies.append(start.elapsed_time(end))

    avg_us = sum(latencies) / len(latencies) / graph_loop * 1000
    graph.reset()
    return avg_us


def bench_cuda_graph_symm(
    func,
    warmup: int,
    graph_loop: int,
    test_loop: int,
) -> float:
    """Benchmark inside a CUDA graph with symmetric memory context."""
    with SymmetricMemoryContext(get_tensor_model_parallel_group()):
        pass

    with graph_capture() as graph_capture_context:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=graph_capture_context.stream):
            for _ in range(graph_loop):
                func()

    graph.replay()

    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    latencies: List[float] = []
    for _ in range(test_loop):
        torch.cuda.synchronize()
        dist.barrier()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        latencies.append(start.elapsed_time(end))

    avg_us = sum(latencies) / len(latencies) / graph_loop * 1000
    graph.reset()
    return avg_us


def human_readable_size(size_bytes, decimal_places=1):
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if size_bytes < 1024.0 or unit == "TiB":
            break
        size_bytes /= 1024.0
    return f"{size_bytes:.{decimal_places}f} {unit}"


try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


def print_markdown_table(data, title=""):
    if not data:
        return
    if title:
        print(f"\n{title}")
    if tabulate is not None:
        print(tabulate(data, headers="keys", tablefmt="github", floatfmt=".2f"))
    else:
        headers = list(data[0].keys())
        header_row = "| " + " | ".join(headers) + " |"
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        rows = []
        for item in data:
            row = "| " + " | ".join(str(item[k]) for k in headers) + " |"
            rows.append(row)
        print("\n".join([header_row, separator] + rows))
    print()


def main():
    args = parse_args()

    import logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S", force=True,
    )

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(rank % 8)
    device = torch.cuda.current_device()

    init_distributed_environment(
        world_size=world_size, rank=rank, local_rank=rank % 8,
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    group = get_tensor_model_parallel_group().device_group
    pynccl_comm = get_tensor_model_parallel_group().pynccl_comm
    dist.barrier()

    N = world_size
    H = args.num_heads
    D = args.head_dim
    H_local = H // N
    dtype = torch.bfloat16
    lpd = 2  # bf16: 2 elements to store one fp32 LSE value

    assert H % N == 0, f"num_heads ({H}) must be divisible by world_size ({N})"

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    if IS_CI:
        batch_sizes = batch_sizes[:2]

    if rank == 0:
        print(f"DCP Communication Benchmark")
        print(f"  world_size (N) = {N}")
        print(f"  num_heads  (H) = {H}, heads_per_rank (H_local) = {H_local}")
        print(f"  head_dim   (D) = {D}")
        print(f"  dtype          = {dtype}")
        print(f"  batch_sizes    = {batch_sizes}")
        print()

    eager_results = []
    graph_results = []

    for B in batch_sizes:
        # --- Allocate A2A buffers (fused output+LSE) ---
        send_combined = torch.randn(
            N, B, H_local, D + lpd, dtype=dtype, device=device
        )
        recv_combined = torch.empty_like(send_combined)

        a2a_msg_bytes = send_combined.nbytes

        # --- Allocate AG+RS buffers ---
        attn_out_flat = torch.randn(B * H_local, D, dtype=dtype, device=device)
        attn_lse = torch.randn(B, H, dtype=torch.float32, device=device)
        ag_lse_output = torch.empty(N * B, H, dtype=torch.float32, device=device)
        rs_output = torch.empty(B * H_local // N, D, dtype=dtype, device=device)

        ag_msg_bytes = attn_lse.nbytes * N  # all_gather inflates by N
        rs_msg_bytes = attn_out_flat.nbytes  # reduce_scatter input
        agrs_total_bytes = ag_msg_bytes + rs_msg_bytes

        if rank == 0:
            print(
                f"B={B}: A2A msg={human_readable_size(a2a_msg_bytes)}, "
                f"AG+RS msg={human_readable_size(agrs_total_bytes)} "
                f"(AG LSE={human_readable_size(ag_msg_bytes)}, "
                f"RS out={human_readable_size(rs_msg_bytes)})"
            )

        # --- Eager benchmarks ---
        if not args.skip_eager:
            a2a_eager_us = bench_eager(
                lambda: a2a_fused_comm(send_combined, recv_combined, group),
                args.warmup, args.test_loop,
            )
            agrs_eager_us = bench_eager(
                lambda: ag_rs_comm(
                    attn_out_flat, attn_lse, ag_lse_output, rs_output, group,
                ),
                args.warmup, args.test_loop,
            )

            a2a_pynccl_eager_us = bench_eager(
                lambda: a2a_fused_comm_pynccl(
                    send_combined, recv_combined, pynccl_comm,
                ),
                args.warmup, args.test_loop,
            )
            agrs_pynccl_eager_us = bench_eager(
                lambda: ag_rs_comm_pynccl(
                    attn_out_flat, attn_lse, ag_lse_output, rs_output,
                    pynccl_comm,
                ),
                args.warmup, args.test_loop,
            )

            eager_results.append({
                "batch": B,
                "a2a_msg": human_readable_size(a2a_msg_bytes),
                "agrs_msg": human_readable_size(agrs_total_bytes),
                "[A2A] torch eager (us)": f"{a2a_eager_us:.2f}",
                "[A2A] pynccl eager (us)": f"{a2a_pynccl_eager_us:.2f}",
                "[AG+RS] torch eager (us)": f"{agrs_eager_us:.2f}",
                "[AG+RS] pynccl eager (us)": f"{agrs_pynccl_eager_us:.2f}",
            })

        # --- CUDA graph benchmarks ---
        if not args.skip_graph:
            a2a_graph_us = bench_cuda_graph_symm(
                lambda: a2a_fused_comm_pynccl(
                    send_combined, recv_combined, pynccl_comm,
                ),
                args.warmup, args.graph_loop, args.test_loop,
            )
            agrs_graph_us = bench_cuda_graph_symm(
                lambda: ag_rs_comm_pynccl(
                    attn_out_flat, attn_lse, ag_lse_output, rs_output,
                    pynccl_comm,
                ),
                args.warmup, args.graph_loop, args.test_loop,
            )

            graph_results.append({
                "batch": B,
                "a2a_msg": human_readable_size(a2a_msg_bytes),
                "agrs_msg": human_readable_size(agrs_total_bytes),
                "[A2A] pynccl graph (us)": f"{a2a_graph_us:.2f}",
                "[AG+RS] pynccl graph (us)": f"{agrs_graph_us:.2f}",
            })

        if rank == 0:
            print(f"  B={B} done.")

    if rank == 0:
        if eager_results:
            print_markdown_table(eager_results, "=== Eager Mode ===")
        if graph_results:
            print_markdown_table(graph_results, "=== CUDA Graph Mode ===")


if __name__ == "__main__":
    main()
