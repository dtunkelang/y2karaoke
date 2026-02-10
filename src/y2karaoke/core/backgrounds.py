"""Compatibility module alias for background processing implementation."""

from .components.render import backgrounds as _impl
import sys as _sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .components.render.backgrounds import (  # noqa: F401
        BackgroundProcessor,
        BackgroundSegment,
        create_background_segments,
    )

_sys.modules[__name__] = _impl
