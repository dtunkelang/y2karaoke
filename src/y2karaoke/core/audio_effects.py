"""Compatibility module alias for audio effects implementation."""

from .components.audio import audio_effects as _impl
import sys as _sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .components.audio.audio_effects import AudioProcessor, process_audio

_sys.modules[__name__] = _impl
