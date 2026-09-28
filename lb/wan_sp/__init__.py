"""End-to-end sequence-parallel + load-balanced sparse attention for Wan T2V.

The E2E sibling of ``lb/bench_sp_all2all_attention.py``: the same all2all +
head-placement primitives, wired into the real diffusers Wan denoising loop.
"""

from .context import (
    LayerPlan,
    SPContext,
    SPTiming,
    get_sp_context,
    set_sp_context,
    sp_enabled,
)
from .install import install_wan_sp

__all__ = [
    "LayerPlan",
    "SPContext",
    "SPTiming",
    "get_sp_context",
    "set_sp_context",
    "sp_enabled",
    "install_wan_sp",
]
