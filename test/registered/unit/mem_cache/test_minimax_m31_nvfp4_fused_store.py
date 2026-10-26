"""CPU checks of the M3.1 NVFP4 KV pool's fused stores
(``mem_cache/minimax_m3_nvfp4_pool.py``): ``set_fused_kv_index_buffer`` hands its
kernel what ``_nvfp4_store_kv_index_args`` returns and stores per buffer exactly where
that returns None; ``set_kv_index_buffer_fused_qknorm_rope`` launches the fused q/k
norm + RoPE + store op (``minimax_qknorm_rope_store_fp8``) once on the same store
arguments, the pool's own buffers, and returns True, or launches nothing and returns
False where the store would be per buffer; all three raise alike. The kernels are
faked; they are checked on GPU (``kernels/ops/kvcache/test_minimax_m31_nvfp4.py``,
``kernels/ops/attention/test_minimax_qknorm_rope_store.py``)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention import minimax_qknorm_rope_store
from sglang.kernels.ops.kvcache import nvfp4_store_kv_index
from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import MiniMaxM3Nvfp4KVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

PAGE = 128
NUM_LAYERS = 2
POOL_TOKENS = 5 * PAGE
QKV_WIDTH = 2560  # one TP4 rank's q|k|v|index q|index k row: 20 heads of 128
# MiniMaxM3Attention._qknorm_groups' order: q, k, index q, index k
GROUPS = ((0, 16), (16, 1), (18, 1), (19, 1))


def _pool():
    """SGLang's M3.1 KV pool (NVFP4 main K/V and index K) on the CPU, every byte
    random, with its fused store on (the GPU's setting)."""
    pool = MiniMaxM3Nvfp4KVPool(
        size=POOL_TOKENS,
        page_size=PAGE,
        dtype=torch.float8_e4m3fn,
        head_num=1,
        head_dim=128,
        idx_head_dim=128,
        dense_layer_ids=[],
        sparse_layer_ids=list(range(NUM_LAYERS)),
        disable_value_sparse_layer_ids=list(range(NUM_LAYERS)),
        device="cpu",
        start_layer=0,
        end_layer=NUM_LAYERS,
    )
    gen = torch.Generator().manual_seed(5)
    for slab in (pool.main_pool.slab, pool.index_k_pool.slab):
        slab.copy_(torch.randint(0, 256, slab.shape, dtype=torch.uint8, generator=gen))
    pool.use_nvfp4_fused_quantize_store = True
    return pool


def _rows(rows=4, dtype=torch.bfloat16):
    """q|k|v|index rows and the views the attention hands the store: K, V
    ``[T, 1, 128]`` and index K ``[T, 1, 128]``, in place in the rows."""
    qkv = torch.randn(rows, QKV_WIDTH).to(dtype)
    k = qkv[:, 2048:2176].view(rows, 1, 128)
    v = qkv[:, 2176:2304].view(rows, 1, 128)
    idx_k = qkv[:, 2432:].reshape(rows, 1, 128)
    return qkv, k, v, idx_k


def _norm(qkv):
    """The q/k norm's arguments: (groups, cos_sin_cache, positions, eps) and its FP8
    outputs and groups."""
    weights = [torch.ones(128, dtype=torch.bfloat16) for _ in GROUPS]
    groups = [(w, off, cnt) for w, (off, cnt) in zip(weights, GROUPS)]
    rows = qkv.shape[0]
    fp8 = dict(
        fp8_a=torch.empty(rows, 16, 128, dtype=torch.float8_e4m3fn),
        group_a=0,
        fp8_b=torch.empty(rows, 1, 128, dtype=torch.float8_e4m3fn),
        group_b=2,
    )
    return (groups, torch.zeros(64, 64), torch.arange(rows), 1e-6), fp8


def _layout(t):
    return (t.data_ptr(), tuple(t.shape), t.stride(), t.dtype)


def _assert_same_args(case, got, want):
    case.assertEqual(got.keys(), want.keys())
    for name, value in want.items():
        if isinstance(value, torch.Tensor):
            case.assertEqual(_layout(got[name]), _layout(value), name)
        else:
            case.assertEqual(got[name], value, name)


def _fail(name):
    def fail(*args, **kwargs):
        raise AssertionError(f"{name} must not run")

    return fail


class TestPoolStoreArguments(CustomTestCase):
    """set_fused_kv_index_buffer hands its kernel what _nvfp4_store_kv_index_args
    returns, stores per buffer exactly where that returns None, and both raise
    alike."""

    def test_the_store_and_its_arguments(self):
        pool = _pool()
        layer = SimpleNamespace(layer_id=1)
        _, k, v, idx_k = _rows()
        loc = torch.tensor([3, 130, 255, 0])
        calls = []
        with patch.object(
            nvfp4_store_kv_index,
            "nvfp4_store_kv_index",
            lambda *a, **kw: calls.append((a, kw)),
        ):
            pool.set_fused_kv_index_buffer(layer, loc, k, v, idx_k, None)
        args = pool._nvfp4_store_kv_index_args(layer, loc, k, v, idx_k, None)
        ((call_args, call_kwargs),) = calls
        self.assertEqual(call_args, ())
        _assert_same_args(self, call_kwargs, args)
        main = pool.main_pool.get_raw_kv_buffers(1)
        self.assertIs(args["k_payload"], main[0])
        self.assertEqual(args["idx_k"].data_ptr(), idx_k.data_ptr())
        self.assertEqual(args["page_size"], PAGE)
        pool.use_nvfp4_fused_quantize_store = False
        self.assertIsNone(
            pool._nvfp4_store_kv_index_args(layer, loc, k, v, idx_k, None)
        )
        for bad in (dict(cache_idx_v=idx_k), dict(k_scale=2.0)):
            kw = dict(cache_idx_v=None) | bad
            with self.assertRaises(ValueError):
                pool._nvfp4_store_kv_index_args(layer, loc, k, v, idx_k, **kw)
            with self.assertRaises(ValueError):
                pool.set_fused_kv_index_buffer(layer, loc, k, v, idx_k, **kw)

    def test_per_buffer_where_the_args_are_none(self):
        pool = _pool()
        layer = SimpleNamespace(layer_id=0)
        _, k, v, idx_k = _rows()
        loc = torch.tensor([1, 2, 3, 4])
        pool.use_nvfp4_fused_quantize_store = False
        with (
            patch.object(
                nvfp4_store_kv_index,
                "nvfp4_store_kv_index",
                _fail("nvfp4_store_kv_index"),
            ),
            patch.object(pool, "set_kv_buffer") as kv,
            patch.object(pool, "set_index_k_buffer") as index,
        ):
            pool.set_fused_kv_index_buffer(layer, loc, k, v, idx_k, None)
        kv.assert_called_once_with(layer, loc, k, v, None, None)
        index.assert_called_once_with(layer, loc, idx_k, None)


class TestFusedQkNormRopeStore(CustomTestCase):
    """set_kv_index_buffer_fused_qknorm_rope: the fused op on the q/k norm's
    arguments and the store's (_nvfp4_store_kv_index_args, the pool's own buffers),
    once; nothing where the store would be per buffer."""

    def _call(self, pool, layer, loc, qkv, k, v, idx_k, **kw):
        norm, fp8 = _norm(qkv)
        return pool.set_kv_index_buffer_fused_qknorm_rope(
            layer,
            loc,
            qkv,
            *norm,
            **fp8,
            k_group=1,
            index_k_group=3,
            cache_k=k,
            cache_v=v,
            cache_idx_k=idx_k,
            **kw,
        ), (norm, fp8)

    def test_launches_the_fused_op_on_the_stores_arguments(self):
        pool = _pool()
        layer = SimpleNamespace(layer_id=1)
        qkv, k, v, idx_k = _rows()
        loc = torch.tensor([3, 130, 255, 0])
        before = (pool.main_pool.slab.clone(), pool.index_k_pool.slab.clone())
        calls = []
        with (
            patch.object(
                minimax_qknorm_rope_store,
                "minimax_qknorm_rope_store_fp8",
                lambda *a, **kw: calls.append((a, kw)),
            ),
            patch.object(
                nvfp4_store_kv_index,
                "nvfp4_store_kv_index",
                _fail("nvfp4_store_kv_index"),
            ),
            patch.object(pool, "set_kv_buffer", _fail("set_kv_buffer")),
            patch.object(pool, "set_index_k_buffer", _fail("set_index_k_buffer")),
        ):
            launched, (norm, fp8) = self._call(
                pool, layer, loc, qkv, k, v, idx_k, k_scale=1.0
            )
        self.assertIs(launched, True)
        ((args, kwargs),) = calls
        self.assertIs(args[0], qkv)
        self.assertEqual(args[1:], norm)
        for name, value in fp8.items():
            self.assertIs(kwargs.pop(name), value)
        self.assertEqual((kwargs.pop("k_group"), kwargs.pop("index_k_group")), (1, 3))
        # the store's arguments: what set_fused_kv_index_buffer hands its kernel
        _assert_same_args(
            self, kwargs, pool._nvfp4_store_kv_index_args(layer, loc, k, v, idx_k, None)
        )
        main = pool.main_pool.get_raw_kv_buffers(1)
        index = pool.index_k_pool.get_raw_k_buffers(pool.index_k_layer_id_mapping[1])
        for name, buffer in zip(("k_payload", "v_payload", "k_scale", "v_scale"), main):
            self.assertIs(kwargs[name], buffer, name)
        self.assertIs(kwargs["idx_k_payload"], index[0])
        self.assertIs(kwargs["idx_k_scale"], index[1])
        self.assertEqual(kwargs["k"].data_ptr(), k.data_ptr())
        self.assertEqual(kwargs["idx_k"].data_ptr(), idx_k.data_ptr())
        # the faked kernel wrote nothing, and nothing else did
        self.assertTrue(torch.equal(pool.main_pool.slab, before[0]))
        self.assertTrue(torch.equal(pool.index_k_pool.slab, before[1]))

    def test_nothing_where_the_store_would_be_per_buffer(self):
        layer = SimpleNamespace(layer_id=0)
        loc = torch.tensor([1, 2, 3, 4])
        for case in ("fused store off", "float16 rows"):
            with self.subTest(case=case):
                pool = _pool()
                if case == "fused store off":
                    pool.use_nvfp4_fused_quantize_store = False
                    qkv, k, v, idx_k = _rows()
                else:
                    qkv, k, v, idx_k = _rows(dtype=torch.float16)
                self.assertIsNone(
                    pool._nvfp4_store_kv_index_args(layer, loc, k, v, idx_k, None)
                )
                before = (pool.main_pool.slab.clone(), pool.index_k_pool.slab.clone())
                with (
                    patch.object(
                        minimax_qknorm_rope_store,
                        "minimax_qknorm_rope_store_fp8",
                        _fail("minimax_qknorm_rope_store_fp8"),
                    ),
                    patch.object(pool, "set_kv_buffer", _fail("set_kv_buffer")),
                    patch.object(
                        pool, "set_index_k_buffer", _fail("set_index_k_buffer")
                    ),
                ):
                    launched, _ = self._call(pool, layer, loc, qkv, k, v, idx_k)
                self.assertIs(launched, False)
                self.assertTrue(torch.equal(pool.main_pool.slab, before[0]))
                self.assertTrue(torch.equal(pool.index_k_pool.slab, before[1]))

    def test_raises_where_the_store_raises(self):
        pool = _pool()
        layer = SimpleNamespace(layer_id=0)
        qkv, k, v, idx_k = _rows()
        loc = torch.tensor([1, 2, 3, 4])
        with patch.object(
            minimax_qknorm_rope_store,
            "minimax_qknorm_rope_store_fp8",
            _fail("minimax_qknorm_rope_store_fp8"),
        ):
            for kw in (dict(k_scale=2.0), dict(v_scale=0.5), dict(idx_k_scale=3.0)):
                with self.subTest(**kw), self.assertRaises(ValueError):
                    self._call(pool, layer, loc, qkv, k, v, idx_k, **kw)
            # K / V of another shape than the pool's heads
            with self.assertRaises(ValueError):
                self._call(pool, layer, loc, qkv, k, v[:, :, :64], idx_k)


if __name__ == "__main__":
    unittest.main()
