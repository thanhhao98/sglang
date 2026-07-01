# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""``DecodeContextParallelStrategy`` — decode context parallel as a CP-v2 strategy.

DCP's owner rule (``pos % dcp_size == dcp_rank``) IS the interleave layout, so we
subclass :class:`InterleaveCPStrategy` to inherit the ``kind = INTERLEAVE`` identity
and a concrete (instantiable) base. But DCP is a different *phase* and *axis*:

* it runs on the separate ``_DCP`` process group (``parallel_state.get_dcp_*``),
  NOT the prefill ``attn_cp`` axis the parent's ``cp_rank`` reads; and
* it applies during *decode*, so ``can_apply`` checks decode mode.

The strategy implements the decode-CP contract declared on
:class:`ContextParallelStrategy` by delegating to the relocated ``layers/cp/dcp``
primitives — it is a behavior-preserving *seam*, not new logic. The prefill methods
inherited from ``InterleaveCPStrategy`` (``shard_hidden_states`` / ``run_attention``
/ ``materialize_full_kv`` / ...) are never called on the decode path.

This module is imported lazily (only by ``init_decode_cp_strategy``), so it may pull
``planner`` (and thus ``server_args``) without adding a load-time edge to the DCP
package init — see the note in ``layers/cp/dcp/__init__.py``.
"""

from __future__ import annotations

from typing import Any, Optional

from sglang.srt.distributed.parallel_state import get_dcp_group
from sglang.srt.layers.cp.dcp.comm import (
    all_gather_q_for_mla_decode,
    cp_lse_ag_out_rs_mha,
    cp_lse_ag_out_rs_mla,
    get_attention_dcp_rank,
    get_attention_dcp_world_size,
)
from sglang.srt.layers.cp.dcp.layout import (
    filter_dcp_local_kv_indices,
    get_dcp_lens,
    update_local_kv_lens_for_dcp,
)
from sglang.srt.layers.cp.dcp.planner import (
    plan_dcp_decode_metadata,
    prepare_decode_context_parallel_metadata,
)
from sglang.srt.layers.cp.interleave import InterleaveCPStrategy


class DecodeContextParallelStrategy(InterleaveCPStrategy):
    """Decode context parallel on the ``_DCP`` group (owner rule pos % N == rank)."""

    name = "decode_context_parallel"

    def supports_decode(self) -> bool:
        return True

    def can_apply(self, num_tokens: int, forward_batch) -> bool:
        # DCP applies during decode (unlike the prefill/extend interleave parent).
        if self.cp_size <= 1:
            return False
        forward_mode = getattr(forward_batch, "forward_mode", None)
        return forward_mode is None or forward_mode.is_decode()

    # -- group accessors (fallback-safe: 1/0 when DCP is disabled) -------------
    def decode_cp_size(self) -> int:
        return get_attention_dcp_world_size()

    def decode_cp_rank(self) -> int:
        return get_attention_dcp_rank()

    def decode_cp_group(self) -> Any:
        return get_dcp_group()

    # -- KV layout -------------------------------------------------------------
    def local_decode_kv_lens(
        self, lens: Any, dcp_size: int, dcp_rank: int, start: Any = None
    ) -> Any:
        return get_dcp_lens(lens, dcp_size, dcp_rank, start)

    def update_local_decode_kv_lens(self, kv_len_arr: Any) -> None:
        update_local_kv_lens_for_dcp(kv_len_arr)

    def shard_decode_kv_indices(self, kv_indices: Any) -> Any:
        return filter_dcp_local_kv_indices(kv_indices=kv_indices)

    # -- metadata --------------------------------------------------------------
    def build_decode_metadata(self, **kwargs: Any) -> Any:
        return prepare_decode_context_parallel_metadata(**kwargs)

    def plan_decode_metadata(self, **kwargs: Any) -> Any:
        return plan_dcp_decode_metadata(**kwargs)

    # -- attention -------------------------------------------------------------
    def gather_decode_query(self, q_nope_out: Any, q_pe: Any) -> Any:
        return all_gather_q_for_mla_decode(q_nope_out, q_pe)

    def merge_decode_attention(
        self,
        cp_attn_out: Any,
        cp_attn_lse: Any,
        cp_group: Any,
        *,
        backend: str,
        return_lse: bool = False,
        ctx: Optional[Any] = None,
    ) -> Any:
        if backend == "mha":
            return cp_lse_ag_out_rs_mha(
                cp_attn_out, cp_attn_lse, cp_group, return_lse=return_lse
            )
        if backend == "mla":
            return cp_lse_ag_out_rs_mla(cp_attn_out, cp_attn_lse, cp_group, ctx=ctx)
        raise ValueError(
            f"unknown decode-CP attention backend {backend!r}; expected 'mha' or 'mla'"
        )
