"""CPU checks of the fused Step-5 DeepSelect top-512 + region pack.

The fused module must compile DeepSelect's device function with the flags its own
module uses (the device function's fp32 index arithmetic depends on --ftz=false
under --use_fast_math), and the default Step-5 selection path must not call it.
"""

import pathlib
import sys
from unittest.mock import patch

import pytest

import sglang
from sglang.kernels.ops.elementwise import deep_select_topk as ds
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BUILD_OPTIONS = (
    "extra_include_paths",
    "extra_dependencies",
    "extra_cflags",
    "extra_cuda_cflags",
)


def test_the_fused_module_builds_with_deep_select_topks_options():
    with patch.object(ds, "load_jit") as load_jit:
        ds._jit_deep_select_topk_module.__wrapped__(512)
        ds._jit_deep_select_topk_region_pack_module.__wrapped__(512)
    topk, fused = load_jit.call_args_list
    for name in BUILD_OPTIONS:
        assert fused.kwargs[name] == topk.kwargs[name], name
    assert {"--use_fast_math", "--ftz=false"} <= set(fused.kwargs["extra_cuda_cflags"])
    assert fused.kwargs["cuda_files"] == ["elementwise/deep_select_topk_pack.cuh"]
    assert fused.kwargs["cuda_wrappers"] == [
        ("topk_region_pack", "DeepSelectTopKRegionPackFp32Kernel<512>::run")
    ]
    assert fused.args[0] != topk.args[0]  # its own module name
    with pytest.raises(RuntimeError, match="Unsupported topk"):
        ds._jit_deep_select_topk_region_pack_module.__wrapped__(1024)


def test_the_default_step5_path_does_not_call_the_fused_op():
    attention = pathlib.Path(sglang.__file__).parent / "srt" / "layers" / "attention"
    for path in (
        attention / "step5" / "region_selection.py",
        attention / "step5" / "indexer.py",
        attention / "step5_backend.py",
    ):
        text = path.read_text()
        assert "step5_select_regions_deepselect_fused" not in text, path
        assert "deep_select_topk_region_pack" not in text, path


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
