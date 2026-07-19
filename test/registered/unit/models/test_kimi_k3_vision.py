from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.vision import (
    prepare_flashinfer_cudnn_vision_attention_metadata,
)
from sglang.srt.models import kimi_k3_vl
from sglang.srt.models.kimi_k3_vl import (
    KimiK3VisionTower,
    MoonViT3dEncoder,
    interpolate_pos_emb,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
def test_kimi_k3_vision_tower_uses_configured_position_interpolation(mode):
    config = SimpleNamespace(
        patch_size=2,
        init_pos_emb_height=2,
        init_pos_emb_width=3,
        init_pos_emb_time=1,
        pos_emb_type="divided_fixed",
        pos_emb_interpolation_mode=mode,
        patch_embed_proj_bias=False,
        merge_kernel_size=(1, 1),
        merge_type="sd2_tpool",
        vt_hidden_size=8,
        vt_num_attention_heads=1,
        vt_num_hidden_layers=0,
        num_hidden_layers=0,
        vt_intermediate_size=16,
        qkv_hidden_size=8,
        norm_type="rmsnorm",
        activation_func="gelu_pytorch_tanh",
        attn_bias=False,
        linear_bias=False,
    )

    tower = KimiK3VisionTower(config)

    assert tower.patch_embed.pos_emb.interpolation_mode == mode


def test_kimi_k3_position_interpolation_uses_contiguous_chw(monkeypatch):
    weight = torch.randn(4, 5, 8)
    output_size = (3, 7)
    expected = (
        F.interpolate(
            weight.permute(2, 0, 1).unsqueeze(0),
            size=output_size,
            mode="bilinear",
        )
        .squeeze(0)
        .permute(1, 2, 0)
        .flatten(end_dim=1)
    )
    original_interpolate = F.interpolate
    captured = {}

    def capture_layout(input_tensor, *args, **kwargs):
        captured["is_contiguous"] = input_tensor.is_contiguous()
        captured["stride"] = input_tensor.stride()
        return original_interpolate(input_tensor, *args, **kwargs)

    monkeypatch.setattr(kimi_k3_vl.F, "interpolate", capture_layout)
    actual = interpolate_pos_emb(weight, "bilinear", output_size)

    assert captured == {
        "is_contiguous": True,
        "stride": (160, 20, 5, 1),
    }
    assert torch.equal(actual, expected)


def test_kimi_k3_prepares_shared_attention_metadata_once(monkeypatch):
    metadata_ids = []
    values_are_contiguous = []

    class FakeAttention(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, q, k, v, *, forward_metadata, **kwargs):
            metadata_ids.append(id(forward_metadata))
            values_are_contiguous.append(v.is_contiguous())
            return q

    monkeypatch.setattr(
        kimi_k3_vl,
        "get_server_args",
        lambda: SimpleNamespace(mm_attention_backend="flashinfer_cudnn"),
    )
    monkeypatch.setitem(
        kimi_k3_vl.QKV_BACKEND_IMPL, "flashinfer_cudnn", FakeAttention
    )

    encoder = MoonViT3dEncoder(
        hidden_dim=8,
        num_layers=2,
        block_cfg={
            "num_heads": 1,
            "hidden_dim": 8,
            "qkv_hidden_size": 8,
            "mlp_dim": 16,
            "norm_type": "rmsnorm",
            "activation": F.gelu,
            "attn_bias": False,
            "linear_bias": False,
        },
    )
    output = encoder(torch.randn(4, 8), torch.tensor([[1, 2, 2]]))

    assert output.shape == (4, 8)
    assert len(metadata_ids) == 2
    assert len(set(metadata_ids)) == 1
    assert values_are_contiguous == [True, True]


def test_flashinfer_cudnn_metadata_uses_bucketed_element_indptrs():
    metadata = prepare_flashinfer_cudnn_vision_attention_metadata(
        torch.tensor([0, 480, 1200], dtype=torch.int32),
        device=torch.device("cpu"),
        elem_per_token=1536,
    )

    expected_indptr = torch.tensor(
        [0, 480 * 1536, 1200 * 1536] + [1200 * 1536] * 6,
        dtype=torch.int32,
    )
    assert torch.equal(metadata.packed_indptrs, expected_indptr.repeat(3))
    assert metadata.sequence_lengths.flatten().tolist() == [480, 720] + [0] * 6
    assert metadata.flashinfer_max_seqlen == 4096


def test_kimi_k3_fused_rope_gate_is_prepared_once_per_encoder(monkeypatch):
    monkeypatch.setattr(kimi_k3_vl, "_can_use_fused_rope", lambda *args: True)
    fused_calls = []

    def fake_fused(q, k, freqs):
        fused_calls.append(q.shape)
        return kimi_k3_vl.apply_rope(q, k, freqs)

    monkeypatch.setattr(kimi_k3_vl, "apply_fused_qk_complex_rope", fake_fused)

    encoder = MoonViT3dEncoder(
        hidden_dim=8,
        num_layers=2,
        block_cfg={
            "num_heads": 1,
            "hidden_dim": 8,
            "qkv_hidden_size": 8,
            "mlp_dim": 16,
            "norm_type": "rmsnorm",
            "activation": F.gelu,
            "attn_bias": False,
            "linear_bias": False,
        },
    )
    output = encoder(torch.randn(4, 8), torch.tensor([[1, 2, 2]]))

    assert output.shape == (4, 8)
    assert fused_calls == [torch.Size([4, 1, 8])] * 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
