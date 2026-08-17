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

"""Pure index math for decode context parallel (DCP): per-rank lengths and
the owner-rule local-index filter."""

import os

import torch

from sglang.srt.runtime_context import get_parallel

# Audit instrumentation (PR #33926 review). Zero-cost unless SGLANG_DCP_AUDIT
# is set in the environment at import time.
_DCP_AUDIT = bool(os.environ.get("SGLANG_DCP_AUDIT"))
_dcp_audit_get_dcp_lens_calls = 0


def get_dcp_audit_lens_call_count() -> int:
    """[DCP_AUDIT] Total get_dcp_lens invocations (finding6 counter)."""
    return _dcp_audit_get_dcp_lens_calls


def get_dcp_lens(
    lens: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    start: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-rank visible KV length under the owner rule pos % dcp_size == dcp_rank.

    Superset implementation (PR #25090): supports both start=None and a per-request
    `start` offset. update_local_kv_lens_for_dcp is the start=None special case.
    """
    if _DCP_AUDIT:
        global _dcp_audit_get_dcp_lens_calls
        _dcp_audit_get_dcp_lens_calls += 1
    if dcp_size == 1:
        return lens
    if start is None:
        return lens // dcp_size + (dcp_rank < lens % dcp_size)

    first = start + torch.remainder(dcp_rank - start, dcp_size)
    remaining = start + lens - first
    return torch.clamp((remaining + dcp_size - 1) // dcp_size, min=0)


def filter_dcp_local_kv_indices(kv_indices: torch.Tensor):
    parallel = get_parallel()
    if parallel.dcp_enabled:
        kv_indices = (
            kv_indices[kv_indices % parallel.dcp_size == parallel.dcp_rank]
            // parallel.dcp_size
        )
    return kv_indices


def update_local_kv_lens_for_dcp(kv_len_arr):
    """In-place per-rank KV length: the start=0 case of get_dcp_lens.

    floor((len - rank - 1) / N) + 1  ==  len // N + (rank < len % N)  for len >= 0
    (bit-identical; see test/registered/cp/test_dcp_layout_unit.py). Kept as an
    in-place mutation because callers (plan_dcp_decode_metadata, the FlashInfer-MLA
    cuda-graph replay path) rely on it.
    """
    parallel = get_parallel()
    if not parallel.dcp_enabled:
        return
    kv_len_arr.copy_(get_dcp_lens(kv_len_arr, parallel.dcp_size, parallel.dcp_rank))
