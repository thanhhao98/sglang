"""Benchmark: MiniMax-M3.1's fused q/k/index Gemma-RMSNorm + partial RoPE + FP8 q and
NVFP4 K/V/index-K page store (minimax_qknorm_rope_store_fp8, 1 launch) vs the two
kernels it replaces (minimax_qknorm_rope_grouped_fp8, then the nvfp4_store_kv_index
that MiniMaxM3Nvfp4KVPool.set_fused_kv_index_buffer runs), on one TP4 rank's heads
(q 16, k / v 1, index q / k 1, of 128) and SGLang's NVFP4 pool of a 1M-token context,
at decode steps (1 row a request) and DSpark verify steps (8 rows a request).
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.attention.minimax_qknorm_rope import (
    minimax_qknorm_rope_grouped_fp8,
)
from sglang.kernels.ops.attention.minimax_qknorm_rope_store import (
    minimax_qknorm_rope_store_fp8,
)
from sglang.kernels.ops.kvcache.nvfp4_store_kv_index import nvfp4_store_kv_index
from sglang.srt.runtime_context import get_platform
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="nightly", runner_config="4-gpu-b200")

EPS = 1e-6  # text_config.rms_norm_eps
ROPE_THETA = 1e7
ROPE_DIM = 64
HEAD_DIM = 128
PAGE = 128
POOL_TOKENS = 1 << 20
# One TP4 rank's q|k|v|index rows, [q 16 heads | k | v | index q | index k] of 128, and
# the groups normed (head offset, head count): q, k, index q, index k; V is not.
NUM_HEADS = 16
NUM_QKV_HEADS = 20
NORM_GROUPS = ((0, 16), (16, 1), (18, 1), (19, 1))
K_HEAD, V_HEAD, INDEX_K_HEAD = 16, 17, 19
# A step's positions index the first POS_WINDOW rows of the RoPE cache, so that every
# rotated input copy has its own; M3.1's 1M-position cache (256 MiB) cannot rotate.
POS_WINDOW = 4096
ROWS = [1, 2, 4, 8, 16, 32, 64, 128]
ROWS_PER_REQUEST = {"decode": 1, "verify": 8}


@cache_once
def _rope_cache() -> torch.Tensor:
    """M3.1's RoPE cache for positions 0 .. POS_WINDOW - 1: fp32 [pos, cos | sin]."""
    dims = torch.arange(0, ROPE_DIM, 2, dtype=torch.float, device="cuda") / ROPE_DIM
    positions = torch.arange(POS_WINDOW, dtype=torch.float, device="cuda")
    freqs = torch.outer(positions, 1.0 / ROPE_THETA**dims)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()


@cache_once
def _pool():
    """The page buffers and the outer scale that MiniMaxM3Nvfp4KVPool's
    set_fused_kv_index_buffer hands its store kernel (one layer of SGLang's M3.1 NVFP4
    pool), and the pool's page count."""
    from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import MiniMaxM3Nvfp4KVPool

    pool = MiniMaxM3Nvfp4KVPool(
        size=POOL_TOKENS,
        page_size=PAGE,
        dtype=torch.float8_e4m3fn,
        head_num=1,
        head_dim=HEAD_DIM,
        idx_head_dim=HEAD_DIM,
        dense_layer_ids=[],
        sparse_layer_ids=[0],
        disable_value_sparse_layer_ids=[0],
        device="cuda",
        start_layer=0,
        end_layer=1,
    )
    main = pool.main_pool
    k_payload, v_payload, k_scale, v_scale = main.get_raw_kv_buffers(0)
    idx_k_payload, idx_k_scale = pool.index_k_pool.get_raw_k_buffers(
        pool.index_k_layer_id_mapping[0]
    )
    pages = dict(
        k_payload=k_payload,
        k_scale=k_scale,
        v_payload=v_payload,
        v_scale=v_scale,
        idx_k_payload=idx_k_payload,
        idx_k_scale=idx_k_scale,
    )
    return pages, main.global_scale, main.num_pages


def _step(rows: int, rows_per_request: int, num_pages: int, gen: torch.Generator):
    """int64 (slots, positions): requests of ``rows_per_request`` rows on consecutive
    slots in a page of their own (not page 0, a decode graph's padding slot) and on
    consecutive positions."""
    requests = rows // rows_per_request
    pages = torch.randperm(num_pages - 1, generator=gen, device="cuda")[:requests] + 1
    offsets = torch.randint(
        0, PAGE - rows_per_request + 1, (requests,), generator=gen, device="cuda"
    )
    starts = torch.randperm(
        POS_WINDOW // rows_per_request, generator=gen, device="cuda"
    )
    starts = starts[:requests] * rows_per_request
    steps = torch.arange(rows_per_request, device="cuda")
    slots = (pages * PAGE + offsets)[:, None] + steps
    positions = starts[:, None] + steps
    return slots.reshape(-1), positions.reshape(-1)


def _kv_views(qkv: torch.Tensor):
    """K and V [T, 1, 128] and index K [T, 128]: heads of ``qkv`` in place."""
    heads = qkv.view(qkv.shape[0], NUM_QKV_HEADS, HEAD_DIM)
    return (
        heads[:, K_HEAD : K_HEAD + 1],
        heads[:, V_HEAD : V_HEAD + 1],
        heads[:, INDEX_K_HEAD],
    )


def _stock(
    qkv,
    groups,
    cos_sin_cache,
    positions,
    slots,
    global_scale,
    *,
    fp8_q,
    fp8_idx_q,
    pages,
):
    minimax_qknorm_rope_grouped_fp8(
        qkv,
        groups,
        cos_sin_cache,
        positions,
        EPS,
        fp8_a=fp8_q,
        group_a=0,
        fp8_b=fp8_idx_q,
        group_b=2,
    )
    k, v, idx_k = _kv_views(qkv)
    nvfp4_store_kv_index(
        k, v, idx_k, **pages, indices=slots, global_scale=global_scale, page_size=PAGE
    )


def _fused(
    qkv,
    groups,
    cos_sin_cache,
    positions,
    slots,
    global_scale,
    *,
    fp8_q,
    fp8_idx_q,
    pages,
):
    k, v, idx_k = _kv_views(qkv)
    minimax_qknorm_rope_store_fp8(
        qkv,
        groups,
        cos_sin_cache,
        positions,
        EPS,
        fp8_a=fp8_q,
        group_a=0,
        fp8_b=fp8_idx_q,
        group_b=2,
        k_group=1,
        index_k_group=3,
        k=k,
        v=v,
        idx_k=idx_k,
        **pages,
        indices=slots,
        global_scale=global_scale,
        page_size=PAGE,
    )


IMPLS = {"stock": _stock, "fused": _fused}


@marker.parametrize("step", list(ROWS_PER_REQUEST))
@marker.parametrize("rows", ROWS, [1, 8, 128])
@marker.benchmark("provider", list(IMPLS))
def benchmark(step: str, rows: int, provider: str):
    rows_per_request = ROWS_PER_REQUEST[step]
    if rows % rows_per_request:
        marker.skip(f"a {step} step has {rows_per_request} rows a request")
    pages, global_scale, num_pages = _pool()
    gen = torch.Generator(device="cuda").manual_seed(rows * 2 + rows_per_request)
    slots, positions = _step(rows, rows_per_request, num_pages, gen)
    qkv = torch.randn(rows, NUM_QKV_HEADS * HEAD_DIM, generator=gen, device="cuda")
    weights = [
        (0.1 * torch.randn(HEAD_DIM, generator=gen, device="cuda")).bfloat16()
        for _ in NORM_GROUPS
    ]
    groups = [(w, offset, count) for w, (offset, count) in zip(weights, NORM_GROUPS)]
    fp8_q = torch.empty(
        rows, NUM_HEADS, HEAD_DIM, dtype=torch.float8_e4m3fn, device="cuda"
    )
    fp8_idx_q = torch.empty(rows, 1, HEAD_DIM, dtype=torch.float8_e4m3fn, device="cuda")
    return marker.do_bench(
        IMPLS[provider],
        input_args=(
            qkv.bfloat16(),
            groups,
            _rope_cache(),
            positions,
            slots,
            global_scale,
        ),
        input_kwargs=dict(fp8_q=fp8_q, fp8_idx_q=fp8_idx_q, pages=pages),
        # Every call reads its own copy of each input (qkv is normed in place); the FP8
        # q and the pool pages are write targets.
        graph_clone_kwargs=None,
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    if not get_platform().is_sm100:
        print("[skip] needs an SM100-family GPU (B200, B300, GB300)")
    else:
        benchmark.run()
