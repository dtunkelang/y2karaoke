"""Compatibility module alias for alignment implementation."""

from .components.alignment import alignment as _impl
import sys as _sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .components.alignment.alignment import (  # noqa: F401
        _apply_adjustments_to_lines,
        adjust_timing_for_duration_mismatch,
        detect_song_start,
    )

_sys.modules[__name__] = _impl
