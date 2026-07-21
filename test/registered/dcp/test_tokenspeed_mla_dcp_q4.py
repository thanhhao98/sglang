import math
import types
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention import tokenspeed_mla_backend as backend_module
from sglang.srt.layers.attention.tokenspeed_mla_backend import TokenspeedMLABackend
from sglang.srt.layers.dcp.comm import (
    get_dcp_a2a_cuda_graph_buffers,
    init_dcp_a2a_cuda_graph_buffers,
)
from sglang.srt.utils import is_tokenspeed_mla_available
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cuda_ci(est_time=600, suite="nightly-1-gpu-b200", nightly=True)


@unittest.skipUnless(
    torch.cuda.is_available() and is_tokenspeed_mla_available(),
    "TokenSpeed MLA requires CUDA and tokenspeed_mla",
)
class TestTokenSpeedMlaDcpQ4(unittest.TestCase):
    def test_cp_causal_q4_matches_unsharded_and_replays(self):
        import tokenspeed_mla

        if torch.cuda.get_device_capability() < (10, 0):
            self.skipTest("TokenSpeed MLA requires Blackwell")

        torch.manual_seed(7)
        device = torch.device("cuda")
        cp_world = 2
        q_len = 4
        num_heads = 8
        kv_lora_rank = 512
        rope_dim = 64
        global_len = 132
        local_len = global_len // cp_world
        scale = 1.0 / math.sqrt(kv_lora_rank + rope_dim)

        global_kv = torch.randn(
            global_len,
            kv_lora_rank + rope_dim,
            dtype=torch.float32,
            device=device,
        )
        # A high-magnitude suffix makes an off-by-one causal bound visible.
        global_kv[-3:, :kv_lora_rank] = torch.tensor(
            [[448.0], [-448.0], [448.0]], device=device
        )
        global_kv = global_kv.to(torch.float8_e4m3fn)
        query = torch.randn(
            1,
            q_len,
            num_heads,
            kv_lora_rank + rope_dim,
            dtype=torch.float32,
            device=device,
        ).to(torch.float8_e4m3fn)
        workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)

        full_cache = torch.zeros(
            3,
            64,
            kv_lora_rank + rope_dim,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        full_cache.view(-1, kv_lora_rank + rope_dim)[:global_len].copy_(global_kv)
        full_out, _ = tokenspeed_mla.tokenspeed_mla_decode(
            query,
            full_cache,
            workspace,
            kv_lora_rank,
            rope_dim,
            torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device),
            torch.tensor([global_len], dtype=torch.int32, device=device),
            global_len,
            scale,
            return_lse=True,
        )

        local_caches = []
        for rank in range(cp_world):
            cache = torch.zeros(
                3,
                32,
                kv_lora_rank + rope_dim,
                dtype=torch.float8_e4m3fn,
                device=device,
            )
            cache.view(-1, kv_lora_rank + rope_dim)[:local_len].copy_(
                global_kv[rank::cp_world]
            )
            local_caches.append(cache)
        local_block_table = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
        local_seq_lens = torch.tensor([local_len], dtype=torch.int32, device=device)
        global_causal_lens = torch.tensor(
            [global_len], dtype=torch.int32, device=device
        )

        def run_cp():
            partials = [
                tokenspeed_mla.tokenspeed_mla_decode(
                    query,
                    local_caches[rank],
                    workspace,
                    kv_lora_rank,
                    rope_dim,
                    local_block_table,
                    local_seq_lens,
                    local_len,
                    scale,
                    return_lse=True,
                    causal_seqs=global_causal_lens,
                    cp_world=cp_world,
                    cp_rank=rank,
                )
                for rank in range(cp_world)
            ]
            outputs = torch.stack([item[0].float() for item in partials])
            lses = torch.stack([item[1] for item in partials])
            global_lse = torch.logsumexp(lses * math.log(2.0), dim=0) / math.log(2.0)
            weights = torch.exp2(lses - global_lse)
            return (outputs * weights.unsqueeze(-1)).sum(dim=0)

        combined = run_cp()
        torch.testing.assert_close(combined, full_out.float(), rtol=3e-2, atol=3e-2)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = run_cp()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_out, combined, rtol=0, atol=0)


class TestTokenSpeedMlaDcpCall(unittest.TestCase):
    def test_decode_forwards_cp_causal_contract_once(self):
        backend = TokenspeedMLABackend.__new__(TokenspeedMLABackend)
        backend._tokenspeed_workspace = torch.empty(1, dtype=torch.int8)
        backend.kv_lora_rank = 8
        backend.qk_rope_head_dim = 4

        query = torch.empty(2, 4, 8, 12, dtype=torch.float8_e4m3fn)
        kv_cache = torch.empty(3, 32, 12, dtype=torch.float8_e4m3fn)
        block_tables = torch.zeros(2, 1, dtype=torch.int32)
        local_lens = torch.tensor([3, 4], dtype=torch.int32)
        global_lens = torch.tensor([21, 29], dtype=torch.int32)
        expected = (
            torch.empty(2, 4, 8, 8),
            torch.empty(2, 4, 8, dtype=torch.float32),
        )
        decode = Mock(return_value=expected)
        fake_tokenspeed = types.SimpleNamespace(tokenspeed_mla_decode=decode)
        layer = types.SimpleNamespace(scaling=0.125, k_scale_float=1.0)
        parallel = types.SimpleNamespace(attn_dcp_size=8, attn_dcp_rank=3)

        with (
            patch.object(
                backend_module, "tokenspeed_mla", fake_tokenspeed, create=True
            ),
            patch.object(backend_module, "get_parallel", return_value=parallel),
            patch.object(backend_module, "is_arch_support_pdl", return_value=False),
        ):
            got = backend._run_decode_kernel(
                query,
                kv_cache,
                block_tables,
                local_lens,
                4,
                layer,
                dcp_causal_seqs=global_lens,
                return_lse=True,
            )

        self.assertIs(got, expected)
        decode.assert_called_once()
        kwargs = decode.call_args.kwargs
        self.assertIs(kwargs["causal_seqs"], global_lens)
        self.assertEqual(kwargs["cp_world"], 8)
        self.assertEqual(kwargs["cp_rank"], 3)
        self.assertTrue(kwargs["causal_mask"])
        self.assertTrue(kwargs["return_lse"])
        self.assertIs(kwargs["block_tables"], block_tables)
        self.assertIs(kwargs["seq_lens"], local_lens)


class TestDcpA2AGraphBuffers(unittest.TestCase):
    def test_decode_and_verify_shapes_are_preallocated_exactly(self):
        group = types.SimpleNamespace(world_size=8)
        init_dcp_a2a_cuda_graph_buffers(
            row_counts=[2, 8],
            num_heads=64,
            head_dim=512,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            cp_group=group,
        )

        decode = get_dcp_a2a_cuda_graph_buffers(
            2, 64, 512, torch.bfloat16, torch.device("cpu")
        )
        verify = get_dcp_a2a_cuda_graph_buffers(
            8, 64, 512, torch.bfloat16, torch.device("cpu")
        )
        self.assertEqual(decode["send_combined"].shape, (8, 2, 8, 514))
        self.assertEqual(verify["recv_combined"].shape, (8, 8, 8, 514))
        self.assertEqual(verify["send_lse"].shape, (8, 8, 8))
        self.assertIsNone(
            get_dcp_a2a_cuda_graph_buffers(
                4, 64, 512, torch.bfloat16, torch.device("cpu")
            )
        )


if __name__ == "__main__":
    unittest.main()
