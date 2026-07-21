"""DCP isolation for speculative draft-model forwards."""

import contextlib

from sglang.srt.runtime_context import get_parallel


def draft_forward_guard(is_draft: bool):
    """Disable DCP for replicated draft-model forwards."""
    if not is_draft:
        return contextlib.nullcontext()
    return get_parallel().override(dcp_enabled=False)
