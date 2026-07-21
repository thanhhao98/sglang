from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def apply_kimi_k3_spec_backend_defaults(server_args: ServerArgs) -> None:
    """Kimi-K3 speculative-decoding backend defaults (explicit flags win).

    Must run before _handle_linear_attn_backend so the triton fill preempts
    its SM100+ bf16-ssm switch of KDA decode to flashinfer.
    """
    from sglang.srt.utils import is_sm100_supported

    if server_args.speculative_algorithm is None:
        return

    # Keep KDA target-verify on triton: the KDA dispatcher routes verify with
    # decode, and the flashinfer verify kernel is slower on all measured
    # shapes and cannot run ragged/compact layouts (uniform [N,T] signature).
    # Fill only when base is the triton default so an explicit base inherits.
    if (
        server_args.linear_attn_decode_backend is None
        and server_args.linear_attn_backend == "triton"
    ):
        server_args.linear_attn_decode_backend = "triton"
        logger.info(
            "Kimi-K3 with speculative decoding: pinning "
            "--linear-attn-decode-backend to triton (keeps KDA verify on "
            "the triton kernel)."
        )

    # dspark's draft is dense MQA; trtllm_mha avoids flashinfer's blocking
    # per-step host plan. DSPARK-only: other spec algos use MLA-family drafts.
    if (
        server_args.speculative_algorithm == "DSPARK"
        and server_args.speculative_draft_attention_backend is None
        and is_sm100_supported()
    ):
        server_args.speculative_draft_attention_backend = "trtllm_mha"
        logger.info(
            "Kimi-K3 DSPARK: defaulting "
            "--speculative-draft-attention-backend to trtllm_mha."
        )
