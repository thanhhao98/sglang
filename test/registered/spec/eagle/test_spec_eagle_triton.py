"""triton attention backend, EAGLE3 chain drafting.

triton runs everywhere, so this stays on the cheap (5090) runner. triton tree
verify is covered by attention/unittests/dense/test_triton.py, and the tree
accept-path compaction e2e lives in test_spec_eagle_topk.py.
"""

import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.matched_stop_kit import MatchedStopMixin
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecFeatureKit,
    SpecLogprobKit,
    SpecPenaltyKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base

register_cuda_ci(est_time=460, stage="base-b", runner_config="1-gpu-small")


class TestEagle3Triton(
    Eagle3Base,
    MatchedStopMixin,
    SpecAccuracyKit,
    SpecLogprobKit,
    SpecPenaltyKit,
    SpecFeatureKit,
):
    """Overlap scheduler on triton (kits listed in bases)."""

    attention_backend = "triton"
    gsm8k_num_examples = 200
    gsm8k_check_accept_len = False
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagle3TritonOverlapPlanStream(TestEagle3Triton):
    """Same config with the overlap plan stream on.

    Nothing else in CUDA CI runs EAGLE under SGLANG_ENABLE_OVERLAP_PLAN_STREAM,
    which is how the plan stream came to read schedule-stream state (seq_lens,
    req_to_token) and overwrite the previous step's captured verify buffers with
    no entry fence. The symptoms are an async illegal memory access and a lower
    accept length, so the detectors are server death, the async OOB probe on the
    verify input_ids, and the accept-length floor turned back on here.

    triton is deliberate: it already implements the post-draft verify fixup
    hook, so this arm isolates the ordering fix from the backend fix.
    """

    gsm8k_check_accept_len = True
    env_overrides = (
        (envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM, True),
        (envs.SGLANG_ENABLE_ASYNC_ASSERT, True),
    )


if __name__ == "__main__":
    unittest.main()
