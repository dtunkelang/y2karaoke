"""Compatibility module alias for audio utility implementation."""

from .components.audio import audio_utils as _impl
import sys as _sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .components.audio.audio_utils import (
        apply_audio_effects,
        separate_vocals,
        trim_audio_if_needed,
    )

_sys.modules[__name__] = _impl
