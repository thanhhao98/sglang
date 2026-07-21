import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

register_cuda_ci(est_time=6000, suite="nightly-8-gpu-b200", nightly=True)

MODEL_PATH = "nvidia/Kimi-K2.5-NVFP4"
DRAFT_MODEL_PATH = "lightseekorg/kimi-k2.5-eagle3-mla"

COMMON_ARGS = [
    "--trust-remote-code",
    "--attention-backend=tokenspeed_mla",
    "--moe-runner-backend=flashinfer_trtllm",
    "--quantization=modelopt_fp4",
    "--kv-cache-dtype=fp8_e4m3",
    "--page-size=64",
    "--mem-fraction-static=0.85",
    "--max-running-requests=16",
    "--speculative-algorithm=EAGLE3",
    f"--speculative-draft-model-path={DRAFT_MODEL_PATH}",
    "--speculative-num-steps=3",
    "--speculative-eagle-topk=1",
    "--speculative-num-draft-tokens=4",
    "--speculative-draft-model-quantization=unquant",
]


class TestKimiK25Nvfp4EagleDcp(unittest.TestCase):
    """Qualify uniform-chain Kimi K2.5 EAGLE3 with TP8/DCP8."""

    def test_kimi_k25_nvfp4_eagle_dcp(self):
        variants = [
            ModelLaunchSettings(
                MODEL_PATH,
                tp_size=8,
                extra_args=COMMON_ARGS,
                variant="TP8+EAGLE3",
            ),
            ModelLaunchSettings(
                MODEL_PATH,
                tp_size=8,
                extra_args=COMMON_ARGS
                + [
                    "--dcp-size=8",
                    "--dcp-comm-backend=a2a",
                    "--dcp-replicate-q-proj",
                ],
                variant="TP8+DCP8+EAGLE3",
            ),
        ]

        run_combined_tests(
            models=variants,
            test_name="Kimi-K2.5-NVFP4 TP8/DCP8 EAGLE3",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                baseline_accuracy=0.92,
                num_examples=200,
                api="completion",
            ),
            performance_params=PerformanceTestParams(
                batch_sizes=[1, 8, 16],
                spec_accept_length_threshold=2.8,
                profile_dir="performance_profiles_kimi_k25_nvfp4_eagle_dcp",
            ),
        )


if __name__ == "__main__":
    unittest.main()
