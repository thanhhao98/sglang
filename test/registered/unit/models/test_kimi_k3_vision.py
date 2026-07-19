from types import SimpleNamespace

import pytest

from sglang.srt.models.kimi_k3_vl import KimiK3VisionTower
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
        vt_intermediate_size=16,
        qkv_hidden_size=8,
        norm_type="rmsnorm",
        activation_func="gelu_pytorch_tanh",
        attn_bias=False,
        linear_bias=False,
    )

    tower = KimiK3VisionTower(config)

    assert tower.patch_embed.pos_emb.interpolation_mode == mode


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
