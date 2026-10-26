"""The fused MiniMax-M3.1 q/k/index norm + RoPE + FP8 q + NVFP4 K/V/index-K store
(``minimax_qknorm_rope_store_fp8``) on one SM100-family GPU (GB300, B300, B200), on
the buffers of SGLang's NVFP4 KV pool and the arguments its fused store hands its
kernel (``MiniMaxM3Nvfp4KVPool.set_fused_kv_index_buffer``), against SGLang's two
kernels as the attention runs them (``minimax_qknorm_rope_grouped_fp8`` in
``forward_prepare``, then the NVFP4 backend's ``set_fused_kv_index_buffer``, i.e.
``nvfp4_store_kv_index``), on one TP4 rank's shapes (q 16 heads, k / v 1 head, index q
/ k 1 head of 128; NeoX RoPE on 64 of 128 dims, theta 1e7), at every decode-step width
(1-16 and 8-128 rows step 8):

- bit for bit: the q|k|v|index rows normed in place, the FP8 q and index q, and every
  byte of SGLang's KV pool for a 1M-token context (main K / V payloads and block
  scales, index-K payload and scales), both pools starting from the same random
  bytes: verify steps of 8-row requests on consecutive cache slots (one crossing a
  page boundary, one on the pool's last slots), decode steps of 1-row requests;
  positions from 0 to the 1M range's last, crossing a page of positions; padded rows
  at slot 0 and position 0 as a decode graph pads them; int64 and int32 slots;
- padded rows whose values differ race on slot 0, in SGLang's store as in the fused
  kernel: every other byte matches, and each 16-value block at slot 0 holds one of the
  padded rows' bytes;
- a CUDA graph of the fused launch, replayed on new rows, positions and slots,
  against the two kernels run eagerly;
- PDL: the fused launch right after a predecessor (compiled here, PDL_SOURCE) that
  lets it launch at once and writes its q|k|v|index rows and cache slots ~50 us later,
  as a GEMM that signals its dependent before its own wait does: eagerly and in graph
  replay, bit for bit with the two kernels on the inputs in place; with a control that
  a kernel launched with PDL that does not wait reads the predecessor's input early
  (else that check would prove nothing);
- (printed, nothing checked) the time a layer takes with the two kernels and with the
  fused one, in CUDA graphs of 60 layers, each followed by a kernel launched without
  PDL as the MSA-v1 indexer chain is.

Run inside the serving container (one GPU):
    python3 -m pytest -q -s test/registered/kernels/ops/attention/test_minimax_qknorm_rope_store.py
"""

import hashlib
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

EPS = 1e-6  # text_config.rms_norm_eps
ROPE_THETA = 1e7
ROPE_DIM = 64
PAGE = 128
MAX_POS = 1 << 20  # the 1M context: RoPE positions 0 .. MAX_POS - 1
POOL_TOKENS = 1 << 20
QKV_WIDTH = 2560  # [q 16 heads | k | v | index q | index k] of 128
NORM_GROUPS = ((0, 16), (16, 1), (18, 1), (19, 1))
WIDTHS = sorted(set(range(1, 17)) | set(range(8, 129, 8)))
INT32_WIDTHS = (1, 8, 16, 64, 128)
GRAPH_WIDTHS = (1, 8, 64, 128)
TIME_WIDTHS = (1, 8, 64, 128)
CHAIN = 60  # layers in a timed graph
TIME_REPLAYS = 20
PDL_WIDTHS = (1, 8, 64, 128)
PDL_SPIN_NS = 50_000  # the predecessor's spin; a PDL launch takes a few microseconds
PDL_REPLAYS = 3
POISON = 0x7F  # bf16 0x7F7F (3.4e38) in the rows; the control's marker byte
FILL = 0x5A  # what an output holds before a replay
# The PDL check's kernels. The predecessor stands in for the q|k|v|index GEMM before
# the fused kernel (a GEMM that lets its dependent launch before it waits, as
# FlashInfer's low-latency MXFP8 GEMM does): launched without PDL, it lets its
# dependents launch at its entry, spins, then writes dst <- src. The control,
# launched with PDL, copies without waiting.
PDL_SOURCE = r"""
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>

namespace sglang::qknorm_rope_store_pdl_test {

namespace {

constexpr int32_t kThreads = 256;

__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void early_copy_kernel(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, int64_t n, uint64_t spin_ns) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const uint64_t start = globaltimer();
  while (globaltimer() - start < spin_ns) {
  }
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    dst[i] = src[i];
  }
}

__global__ void unwaited_copy_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, int64_t n) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    dst[i] = src[i];
  }
}

// One resident wave: every CTA starts, and lets the dependents launch, at once.
int32_t one_wave(int64_t n, int32_t device_id) {
  int32_t sm_count = 0;
  host::RuntimeDeviceCheck(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
  const int64_t blocks = (n + kThreads - 1) / kThreads;
  return static_cast<int32_t>(std::max<int64_t>(1, std::min<int64_t>(blocks, int64_t{sm_count} * 4)));
}

}  // namespace

inline void early_copy(const tvm::ffi::TensorView src, const tvm::ffi::TensorView dst, const int64_t spin_ns) {
  using namespace host;
  auto N = SymbolicSize{"bytes"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({N}).with_dtype<uint8_t>().with_device(device).verify(src).verify(dst);
  const int64_t n = N.unwrap();
  LaunchKernel(dim3(one_wave(n, device.unwrap().device_id)), dim3(kThreads), device.unwrap())(
      early_copy_kernel,
      static_cast<const uint8_t*>(src.data_ptr()),
      static_cast<uint8_t*>(dst.data_ptr()),
      n,
      static_cast<uint64_t>(spin_ns));
}

inline void unwaited_copy(const tvm::ffi::TensorView src, const tvm::ffi::TensorView dst) {
  using namespace host;
  auto N = SymbolicSize{"bytes"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({N}).with_dtype<uint8_t>().with_device(device).verify(src).verify(dst);
  const int64_t n = N.unwrap();
  LaunchKernel(dim3(one_wave(n, device.unwrap().device_id)), dim3(kThreads), device.unwrap())
      .enable_pdl(true)(
          unwaited_copy_kernel, static_cast<const uint8_t*>(src.data_ptr()), static_cast<uint8_t*>(dst.data_ptr()), n);
}

}  // namespace sglang::qknorm_rope_store_pdl_test
"""


def _skip_reason():
    if not torch.cuda.is_available():
        return "needs CUDA"
    if torch.cuda.get_device_capability()[0] != 10:
        return "needs an SM100-family GPU (B200, B300, GB300)"
    from sglang.srt.runtime_context import get_platform

    if not get_platform().is_sm100:
        return "SGLang's fused NVFP4 KV store runs on SM100 / SM103 only"
    return None


_SKIP = _skip_reason()
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")


# --------------------------------------------------------------------- the setup


def _rope_cache():
    """M3.1's RoPE cache as SGLang builds it, over the 1M range: float32
    [MAX_POS, cos 32 | sin 32]."""
    dims = torch.arange(0, ROPE_DIM, 2, dtype=torch.float, device="cuda") / ROPE_DIM
    inv_freq = 1.0 / ROPE_THETA**dims
    freqs = torch.outer(
        torch.arange(MAX_POS, dtype=torch.float, device="cuda"), inv_freq
    )
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()


class _Attention:
    """The MiniMaxM3Attention fields of one TP4 rank that its q/k norm and KV store
    read; its RadixAttention's fields the backend's store reads."""

    num_heads, num_kv_heads, head_dim = 16, 1, 128
    num_idx_heads, idx_head_dim = 1, 128
    q_size, kv_size = 2048, 128
    _fused_main_size = 2304
    disable_index_value = True
    _qknorm_group_meta = NORM_GROUPS

    def __init__(self, cache, seed=7):
        from sglang.srt.models.minimax_m3 import MiniMaxM3Attention

        gen = torch.Generator(device="cuda").manual_seed(seed)
        for name in ("q_norm", "k_norm", "index_q_norm", "index_k_norm"):
            weight = 0.1 * torch.randn(128, device="cuda", generator=gen)
            setattr(
                self,
                name,
                SimpleNamespace(weight=weight.bfloat16(), variance_epsilon=EPS),
            )
        self.rotary_emb = SimpleNamespace(cos_sin_cache=cache)
        self.attn = SimpleNamespace(
            layer_id=0,
            tp_k_head_num=1,
            qk_head_dim=128,
            tp_v_head_num=1,
            v_head_dim=128,
            k_scale_float=None,
            v_scale_float=None,
            idx_k_scale_float=None,
        )
        self._qknorm_groups = MiniMaxM3Attention._qknorm_groups.__get__(self)
        self._split_index_qkv = MiniMaxM3Attention._split_index_qkv.__get__(self)


def _pool():
    """SGLang's M3.1 KV pool (NVFP4 main K / V and index K) for one layer of a
    1M-token context."""
    from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import MiniMaxM3Nvfp4KVPool

    pool = MiniMaxM3Nvfp4KVPool(
        size=POOL_TOKENS,
        page_size=PAGE,
        dtype=torch.float8_e4m3fn,
        head_num=1,
        head_dim=128,
        idx_head_dim=128,
        dense_layer_ids=[],
        sparse_layer_ids=[0],
        disable_value_sparse_layer_ids=[0],
        device="cuda",
        start_layer=0,
        end_layer=1,
    )
    if not pool.use_nvfp4_fused_quantize_store:
        pytest.skip("SGLang's fused NVFP4 KV store is off here")
    return pool


def _slabs(pool):
    return (pool.main_pool.slab, pool.index_k_pool.slab)


def _randomize(pool, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    for slab in _slabs(pool):
        slab.copy_(
            torch.randint(
                0, 256, slab.shape, dtype=torch.uint8, device="cuda", generator=gen
            )
        )


def _copy_pool(dst, src):
    for d, s in zip(_slabs(dst), _slabs(src)):
        d.copy_(s)


@pytest.fixture(scope="module")
def setup():
    cache = _rope_cache()
    pools = {"SGLang": _pool(), "fused": _pool()}
    _randomize(pools["SGLang"], seed=3)
    _copy_pool(pools["fused"], pools["SGLang"])
    return SimpleNamespace(cache=cache, attn=_Attention(cache), pools=pools)


def _total_slots(pool):
    return pool.main_pool.num_pages * PAGE


def _batch(rows, *, tokens, total_slots, gen, padded=0):
    """A step of ``rows`` rows: requests of ``tokens`` rows on consecutive slots and
    positions, each in its own pair of pages (none in page 0: slot 0 is the padded
    rows'). The first request crosses a page boundary of slots (its pair's middle) and
    of positions, the second takes the last positions of the 1M range, the third
    position 0 and the pool's last slots (its last page, in no pair); ``padded``
    trailing rows at slot 0 and position 0. Returns int64 (slots, positions)."""
    real = rows - padded
    requests = -(-real // tokens)
    pairs = total_slots // (2 * PAGE)  # pages 2p, 2p + 1 for p in 1 .. pairs - 1
    pair = torch.randperm(pairs - 1, generator=gen, device="cuda")[:requests] + 1
    offset = torch.randint(
        0, 2 * PAGE - tokens + 1, (requests,), device="cuda", generator=gen
    )
    offset[0] = PAGE - tokens // 2 - 1
    start = pair * 2 * PAGE + offset
    position = torch.randint(
        0, MAX_POS - tokens + 1, (requests,), device="cuda", generator=gen
    )
    position[0] = MAX_POS - PAGE - 3
    if requests > 1:
        position[1] = MAX_POS - tokens
    if requests > 2:
        start[2], position[2] = total_slots - tokens, 0
    steps = torch.arange(tokens, device="cuda")
    loc = (start[:, None] + steps).reshape(-1)[:real]
    pos = (position[:, None] + steps).reshape(-1)[:real]
    zeros = torch.zeros(padded, dtype=torch.int64, device="cuda")
    return torch.cat([loc, zeros]), torch.cat([pos, zeros])


def _check_rows(rows: int, width: int, gen: torch.Generator) -> torch.Tensor:
    """BF16 q|k|v|index rows: Gaussian rows of magnitudes 1e-3 to 1e3, and from row 1
    on rows that reach the store's other branches: all zeros (zero block maxima), one
    value throughout, one large outlier, values below the E4M3 scale's underflow."""
    device = gen.device
    x = torch.randn(rows, width, generator=gen, device=device)
    x *= torch.logspace(-3, 3, rows, device=device)[:, None]
    special = (
        torch.zeros(width, device=device),
        torch.full((width,), -1.5, device=device),
        torch.zeros(width, device=device).index_fill_(
            0, torch.arange(7, width, 97, device=device), 3.0e4
        ),
        torch.full((width,), 1e-4, device=device),
    )
    for i, row in enumerate(special[: rows - 1]):
        x[i + 1] = row
    return x.to(torch.bfloat16)


def _rows(rows, gen, padded=0, *, same_padding=True):
    """q|k|v|index rows as the qkv GEMM leaves them (``_check_rows``' mix of
    magnitudes and special rows); padded rows alike unless ``same_padding`` is
    False."""
    x = _check_rows(rows, QKV_WIDTH, gen)
    if padded and same_padding:
        x[rows - padded :] = x[rows - padded]
    return x


def _views(attn, qkv):
    """forward_prepare's splits and forward_core's views: k / v [T, 1, 128], index k
    [T, 1, 128]."""
    rows = qkv.shape[0]
    _, k, v = qkv[:, : attn._fused_main_size].split([2048, 128, 128], dim=-1)
    _, idx_k, _ = attn._split_index_qkv(qkv[:, attn._fused_main_size :])
    return (
        k.view(rows, 1, 128),
        v.view(rows, 1, 128),
        idx_k.reshape(rows, 1, 128),
    )


def _fp8_outputs(qkv):
    rows = qkv.shape[0]
    e4m3 = torch.float8_e4m3fn
    return (
        qkv.new_empty((rows, 16, 128), dtype=e4m3),
        qkv.new_empty((rows, 1, 128), dtype=e4m3),
    )


def _sglang(attn, pool, qkv, positions, loc):
    """SGLang's attention up to its store: the q/k norm kernel, then the backend's
    store call (MiniMaxSparseNvfp4AttnBackend._forward). Returns the FP8 q, index q."""
    from sglang.kernels.ops.attention.minimax_qknorm_rope import (
        minimax_qknorm_rope_grouped_fp8,
    )

    fp8_q, fp8_idx_q = _fp8_outputs(qkv)
    minimax_qknorm_rope_grouped_fp8(
        qkv,
        attn._qknorm_groups(),
        attn.rotary_emb.cos_sin_cache,
        positions,
        EPS,
        fp8_a=fp8_q,
        group_a=0,
        fp8_b=fp8_idx_q,
        group_b=2,
    )
    k, v, idx_k = _views(attn, qkv)
    layer = attn.attn
    pool.set_fused_kv_index_buffer(
        layer,
        loc,
        k.view(-1, 1, 128),
        v.view(-1, 1, 128),
        idx_k,
        None,
        layer.k_scale_float,
        layer.v_scale_float,
        layer.idx_k_scale_float,
    )
    return fp8_q, fp8_idx_q


def _store_args(pool, layer, loc, k, v, idx_k):
    """What SGLang's fused KV store hands ``nvfp4_store_kv_index`` for these rows
    (``MiniMaxM3Nvfp4KVPool.set_fused_kv_index_buffer``): the K / V / index-K views,
    the layer's raw page buffers, the write slots, the global scale and page size."""
    main, index = pool.main_pool, pool.index_k_pool
    k_payload, v_payload, k_scale, v_scale = main.get_raw_kv_buffers(layer.layer_id)
    idx_payload, idx_scale = index.get_raw_k_buffers(
        pool.index_k_layer_id_mapping[layer.layer_id]
    )
    return dict(
        k=k,
        v=v,
        idx_k=idx_k.reshape(-1, main.head_dim),
        k_payload=k_payload,
        k_scale=k_scale,
        v_payload=v_payload,
        v_scale=v_scale,
        idx_k_payload=idx_payload,
        idx_k_scale=idx_scale,
        indices=loc.reshape(-1),
        global_scale=main.global_scale,
        page_size=main.page_size,
    )


def _fused(attn, pool, qkv, positions, loc):
    """The fused kernel in place of the two, on the store's arguments
    (``_store_args``) and the norm's (as ``_sglang`` passes them). Returns the FP8 q,
    index q."""
    from sglang.kernels.ops.attention.minimax_qknorm_rope_store import (
        minimax_qknorm_rope_store_fp8,
    )

    fp8_q, fp8_idx_q = _fp8_outputs(qkv)
    k, v, idx_k = _views(attn, qkv)
    minimax_qknorm_rope_store_fp8(
        qkv,
        attn._qknorm_groups(),
        attn.rotary_emb.cos_sin_cache,
        positions,
        EPS,
        fp8_a=fp8_q,
        group_a=0,
        fp8_b=fp8_idx_q,
        group_b=2,
        k_group=1,
        index_k_group=3,
        **_store_args(
            pool, attn.attn, loc, k.view(-1, 1, 128), v.view(-1, 1, 128), idx_k
        ),
    )
    return fp8_q, fp8_idx_q


def _bits(t):
    return t.contiguous().view(-1).view(torch.uint8)


def _assert_bitwise(got, want, what):
    assert (got.dtype, tuple(got.shape)) == (want.dtype, tuple(want.shape)), what
    differ = int((_bits(got) != _bits(want)).sum())
    assert differ == 0, f"{what}: {differ} of {_bits(want).numel()} bytes differ"


def _compare(setup, rows, *, tokens, gen, padded=0, loc_dtype=torch.int64):
    """One step through both arms on the same inputs: every output and every pool
    byte bit for bit."""
    attn, pools = setup.attn, setup.pools
    loc, positions = _batch(
        rows,
        tokens=tokens,
        total_slots=_total_slots(pools["SGLang"]),
        gen=gen,
        padded=padded,
    )
    loc = loc.to(loc_dtype)
    x = _rows(rows, gen, padded)
    before = [slab.clone() for slab in _slabs(pools["SGLang"])]
    outs = {}
    for arm, run in (("SGLang", _sglang), ("fused", _fused)):
        qkv = x.clone()
        fp8_q, fp8_idx_q = run(attn, pools[arm], qkv, positions, loc)
        outs[arm] = dict(qkv=qkv, fp8_q=fp8_q, fp8_idx_q=fp8_idx_q)
    torch.cuda.synchronize()
    what = f"{rows} rows ({tokens} a request, {padded} padded, {loc_dtype})"
    for name, want in outs["SGLang"].items():
        _assert_bitwise(outs["fused"][name], want, f"{what}: {name}")
    for name, got, want, old in zip(
        ("main K/V pages", "index-K pages"),
        _slabs(pools["fused"]),
        _slabs(pools["SGLang"]),
        before,
    ):
        _assert_bitwise(got, want, f"{what}: {name}")
        assert not torch.equal(want, old), f"{what}: SGLang's store wrote no {name}"


# ------------------------------------------------------------------------- tests


def test_the_fused_kernel_writes_what_sglangs_two_kernels_write(setup):
    gen = torch.Generator(device="cuda").manual_seed(11)
    for rows in WIDTHS:
        tokens = 8 if rows % 8 == 0 else 1
        _compare(setup, rows, tokens=tokens, gen=gen)
        if tokens == 8:
            _compare(setup, rows, tokens=1, gen=gen)
    for rows in INT32_WIDTHS:
        _compare(setup, rows, tokens=1, gen=gen, loc_dtype=torch.int32)
    print(
        f"minimax_qknorm_rope_store_fp8: bitwise at rows {WIDTHS} (int32 slots at "
        f"{list(INT32_WIDTHS)})"
    )


def test_padded_rows(setup):
    """Padded rows (slot 0, position 0) alike, as a step padded with copies of one
    row: bit for bit, slot 0 included."""
    gen = torch.Generator(device="cuda").manual_seed(13)
    for rows, tokens, padded in ((16, 8, 8), (64, 8, 24), (128, 8, 8), (5, 1, 2)):
        _compare(setup, rows, tokens=tokens, gen=gen, padded=padded)


def _slot0(pool):
    """Slot 0's K, V and index-K payloads (8 blocks of 8 bytes) and block scales (8
    bytes): views into the pool."""
    main, index = pool.main_pool, pool.index_k_pool
    return {
        "k": (main.k_buffer[0][0, 0, 0], main.k_scale_buffer[0][0, 0, 0]),
        "v": (main.v_buffer[0][0, 0, 0], main.v_scale_buffer[0][0, 0, 0]),
        "idx_k": (index.k_buffer[0][0, 0], index.k_scale_buffer[0][0, 0]),
    }


def _slot0_blocks(pool):
    return {
        name: (payload.reshape(8, 8).clone(), scale.reshape(8, 1).clone())
        for name, (payload, scale) in _slot0(pool).items()
    }


def test_padded_rows_that_differ_race_on_slot_0_alike(setup):
    """Padded rows of different values all write slot 0, in SGLang's store as in the
    fused kernel (which write lands is not defined): the outputs and every other byte
    match, and each block at slot 0 holds the bytes one of the padded rows quantizes
    to, in both."""
    attn, pools = setup.attn, setup.pools
    gen = torch.Generator(device="cuda").manual_seed(17)
    rows, padded = 32, 8
    loc, positions = _batch(
        rows,
        tokens=8,
        total_slots=_total_slots(pools["SGLang"]),
        gen=gen,
        padded=padded,
    )
    x = _rows(rows, gen, padded, same_padding=False)
    outs = {}
    for arm, run in (("SGLang", _sglang), ("fused", _fused)):
        qkv = x.clone()
        outs[arm] = (qkv, *run(attn, pools[arm], qkv, positions, loc))
    got = {arm: _slot0_blocks(pool) for arm, pool in pools.items()}
    # each padded row alone at slot 0, through SGLang's store, on a scratch copy
    saved = [slab.clone() for slab in _slabs(pools["SGLang"])]
    candidates = {"k": [], "v": [], "idx_k": []}
    for r in range(rows - padded, rows):
        _sglang(
            attn,
            pools["SGLang"],
            x[r : r + 1].clone(),
            positions[r : r + 1],
            loc[r : r + 1],
        )
        for name, blocks in _slot0_blocks(pools["SGLang"]).items():
            candidates[name].append(blocks)
    for slab, old in zip(_slabs(pools["SGLang"]), saved):
        slab.copy_(old)
    torch.cuda.synchronize()
    for i, name in enumerate(("qkv", "fp8 q", "fp8 index q")):
        _assert_bitwise(outs["fused"][i], outs["SGLang"][i], name)
    for arm, blocks in got.items():
        for name, (payload, scale) in blocks.items():
            for b in range(8):
                assert any(
                    torch.equal(payload[b], cand[0][b]) for cand in candidates[name]
                ), (arm, name, "payload block", b)
                assert any(
                    torch.equal(scale[b], cand[1][b]) for cand in candidates[name]
                ), (arm, name, "block scale", b)
    # every other byte: slot 0 set alike in both pools, then the whole pools
    for pool in pools.values():
        for payload, scale in _slot0(pool).values():
            payload.zero_()
            scale.zero_()
    for name, got_slab, want_slab in zip(
        ("main K/V pages", "index-K pages"),
        _slabs(pools["fused"]),
        _slabs(pools["SGLang"]),
    ):
        _assert_bitwise(got_slab, want_slab, name)


def test_graph_replay(setup):
    """The fused launch captured in a CUDA graph (the qkv GEMM's output copied in
    first), replayed on new rows, positions and slots: bit for bit with the two
    kernels run eagerly."""
    attn, pools = setup.attn, setup.pools
    gen = torch.Generator(device="cuda").manual_seed(19)
    total = _total_slots(pools["SGLang"])
    for rows in GRAPH_WIDTHS:
        tokens = 8 if rows % 8 == 0 else 1
        loc, positions = _batch(rows, tokens=tokens, total_slots=total, gen=gen)
        static_x, static_pos, static_loc = _rows(rows, gen), positions, loc
        work = torch.empty_like(static_x)

        def step():
            work.copy_(static_x)
            return _fused(attn, pools["fused"], work, static_pos, static_loc)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            step()  # warm-up (loads the kernel) off the capture
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fp8_q, fp8_idx_q = step()
        for replay in range(3):
            loc, positions = _batch(rows, tokens=tokens, total_slots=total, gen=gen)
            static_x.copy_(_rows(rows, gen))
            static_pos.copy_(positions)
            static_loc.copy_(loc)
            _copy_pool(pools["fused"], pools["SGLang"])
            graph.replay()
            want_qkv = static_x.clone()
            want_q, want_idx_q = _sglang(
                attn, pools["SGLang"], want_qkv, static_pos, static_loc
            )
            torch.cuda.synchronize()
            what = f"graph at {rows} rows, replay {replay}"
            _assert_bitwise(work, want_qkv, f"{what}: qkv")
            _assert_bitwise(fp8_q, want_q, f"{what}: fp8 q")
            _assert_bitwise(fp8_idx_q, want_idx_q, f"{what}: fp8 index q")
            for name, got, want in zip(
                ("main K/V pages", "index-K pages"),
                _slabs(pools["fused"]),
                _slabs(pools["SGLang"]),
            ):
                _assert_bitwise(got, want, f"{what}: {name}")


@pytest.fixture(scope="module")
def pdl():
    """The PDL check's kernels (PDL_SOURCE), compiled at a path fixed by their source
    (so the JIT cache keeps them)."""
    from sglang.kernels.jit.utils import load_jit

    digest = hashlib.sha256(PDL_SOURCE.encode()).hexdigest()[:16]
    root = Path(tempfile.gettempdir()) / f"sglang_test_qknorm_rope_store_pdl_{digest}"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "early_copy.cuh"
    if not path.exists() or path.read_text() != PDL_SOURCE:
        path.write_text(PDL_SOURCE)
    return load_jit(
        "test_qknorm_rope_store_pdl_early_copy",
        cuda_files=[str(path)],
        cuda_wrappers=[
            ("early_copy", "qknorm_rope_store_pdl_test::early_copy"),
            ("unwaited_copy", "qknorm_rope_store_pdl_test::unwaited_copy"),
        ],
    )


def test_the_predecessor_lets_a_pdl_kernel_start_before_it_writes(pdl):
    """The control: a kernel launched with PDL that copies the predecessor's output
    without waiting reads it before the predecessor writes it, eagerly and in graph
    replay, so the next test can catch a read ahead of griddepcontrol.wait."""
    n = 1 << 20
    src = torch.randint(0, POISON, (n,), dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    seen = torch.empty_like(src)

    def run():
        dst.fill_(POISON)
        pdl.early_copy(src, dst, PDL_SPIN_NS)
        pdl.unwaited_copy(dst, seen)

    # the first launch of each kernel loads its module, which serializes the launches
    # (lazy loading): it would read as "no overlap", so it is not counted
    run()
    torch.cuda.synchronize()
    early = []
    for _ in range(3):
        run()
        torch.cuda.synchronize()
        early.append(float((seen == POISON).float().mean()))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        early.append(float((seen == POISON).float().mean()))
    print(f"QKNORM_ROPE_STORE_PDL_CONTROL poisoned fractions {early}")
    assert all(e > 0.5 for e in early), (
        f"a PDL kernel did not start before the predecessor wrote ({early}): the "
        "PDL check cannot catch a read before griddepcontrol.wait here"
    )
    assert torch.equal(dst, src)


def test_behind_a_predecessor_that_lets_it_start_early(setup, pdl):
    """The fused launch right after the predecessor, which lets it launch at once and
    only ~50 us later writes its q|k|v|index rows (q, K, V, index K) and its cache
    slots (until then poison rows and slot 0, a slot of every pool): a read of any of
    them before the fused kernel's PDL wait would show. Every output and pool byte,
    eagerly and in graph replay, equals SGLang's two kernels on the inputs in place.
    The positions stay a static input, as in serving (both kernels read them before
    their wait). Nothing else may launch between the two kernels, so the async
    asserts (a min / max of the slots) stay off."""
    from sglang.srt.environ import envs

    attn, pools = setup.attn, setup.pools
    gen = torch.Generator(device="cuda").manual_seed(29)
    total = _total_slots(pools["SGLang"])
    with envs.SGLANG_ENABLE_ASYNC_ASSERT.override(False):
        for rows in PDL_WIDTHS:
            tokens = 8 if rows % 8 == 0 else 1
            loc, positions = _batch(rows, tokens=tokens, total_slots=total, gen=gen)
            x = _rows(rows, gen)
            before = [slab.clone() for slab in _slabs(pools["SGLang"])]
            want_qkv = x.clone()
            want_q, want_idx_q = _sglang(
                attn, pools["SGLang"], want_qkv, positions, loc
            )
            # the q|k|v|index rows, then the slots: one buffer the predecessor writes
            row_bytes = x.numel() * x.element_size()
            src = torch.cat([_bits(x), _bits(loc)])
            buf = torch.empty_like(src)
            qkv = buf[:row_bytes].view(torch.bfloat16).view(x.shape)
            slots = buf[row_bytes:].view(torch.int64)

            def step():
                buf[:row_bytes].fill_(POISON)
                buf[row_bytes:].zero_()
                pdl.early_copy(src, buf, PDL_SPIN_NS)
                return _fused(attn, pools["fused"], qkv, positions, slots)

            def reset():
                for slab, old in zip(_slabs(pools["fused"]), before):
                    slab.copy_(old)

            def check(outs, what):
                torch.cuda.synchronize()
                _assert_bitwise(qkv, want_qkv, f"{what}: qkv")
                _assert_bitwise(slots, loc, f"{what}: slots")
                _assert_bitwise(outs[0], want_q, f"{what}: fp8 q")
                _assert_bitwise(outs[1], want_idx_q, f"{what}: fp8 index q")
                for name, got, want in zip(
                    ("main K/V pages", "index-K pages"),
                    _slabs(pools["fused"]),
                    _slabs(pools["SGLang"]),
                ):
                    _assert_bitwise(got, want, f"{what}: {name}")

            for run in range(2):
                reset()
                check(step(), f"eager at {rows} rows, run {run}")
            reset()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outs = step()
            for replay in range(PDL_REPLAYS):
                reset()
                for out in outs:
                    out.view(torch.uint8).fill_(FILL)
                graph.replay()
                check(outs, f"graph at {rows} rows, replay {replay}")
    print(
        "minimax_qknorm_rope_store_fp8: bitwise behind an early-signalling "
        f"predecessor at {PDL_WIDTHS}"
    )


def test_time_per_layer(setup):
    """Printed only: a layer's q/k norm + store with SGLang's two kernels and with
    the fused one, in CUDA graphs of CHAIN layers (each its own q|k|v|index rows),
    each layer followed by a 1-element fill launched without PDL, as the MSA-v1
    indexer chain after the store is."""
    attn, pools = setup.attn, setup.pools
    gen = torch.Generator(device="cuda").manual_seed(23)
    total = _total_slots(pools["SGLang"])
    lines = []
    for rows in TIME_WIDTHS:
        tokens = 8 if rows % 8 == 0 else 1
        loc, positions = _batch(rows, tokens=tokens, total_slots=total, gen=gen)
        layers = [_rows(rows, gen) for _ in range(CHAIN)]
        fence = torch.zeros(1, device="cuda")
        times = {}
        for arm, run in (("SGLang", _sglang), ("fused", _fused)):
            pool = pools[arm]

            def chain():
                for qkv in layers:
                    run(attn, pool, qkv, positions, loc)
                    fence.fill_(1.0)

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                chain()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                chain()
            graph.replay()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(TIME_REPLAYS):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            times[arm] = start.elapsed_time(end) * 1e3 / TIME_REPLAYS / CHAIN
        saved = times["SGLang"] - times["fused"]
        lines.append(
            f"  {rows:3d} rows: two kernels {times['SGLang']:.2f} us, fused "
            f"{times['fused']:.2f} us a layer; {saved:.2f} us a layer, "
            f"{saved * CHAIN:.0f} us a step of {CHAIN} layers"
        )
    _copy_pool(pools["fused"], pools["SGLang"])
    print("QKNORM_ROPE_STORE_TIME (graphs of 60 layers, each layer's rows its own):")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
