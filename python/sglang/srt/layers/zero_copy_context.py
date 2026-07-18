import contextlib
from typing import Iterator, Optional

import msgspec
import torch


class ZeroCopyContext(msgspec.Struct, frozen=True):
    moe_output: Optional[torch.Tensor] = None


ctx = ZeroCopyContext()


@contextlib.contextmanager
def set_moe_output(out: torch.Tensor) -> Iterator[None]:
    """Publish `out` as the MoE runner's output destination for the block."""
    old_output = ctx.moe_output
    msgspec.structs.force_setattr(ctx, "moe_output", out)
    try:
        yield
    finally:
        msgspec.structs.force_setattr(ctx, "moe_output", old_output)


def get_moe_output(ref: torch.Tensor) -> Optional[torch.Tensor]:
    """The published destination iff it can stand in for empty_like(ref)."""
    out = ctx.moe_output
    if (
        out is not None
        and out.shape == ref.shape
        and out.dtype == ref.dtype
        and out.device == ref.device
        and out.is_contiguous()
    ):
        return out
    return None


__all__ = ["ZeroCopyContext", "ctx", "set_moe_output", "get_moe_output"]
