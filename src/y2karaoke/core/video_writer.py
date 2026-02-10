"""Compatibility module alias for video writer implementation."""

from .components.render import video_writer as _impl
import sys as _sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .components.render.video_writer import render_karaoke_video  # noqa: F401

_sys.modules[__name__] = _impl
