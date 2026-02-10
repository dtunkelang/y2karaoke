"""Compatibility module alias for audio separation implementation."""

from .components.audio import separator as _impl
import sys as _sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .components.audio.separator import AudioSeparator, separate_vocals

_sys.modules[__name__] = _impl
