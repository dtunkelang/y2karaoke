"""Compatibility module alias for frame renderer implementation."""

from .components.render import frame_renderer as _impl
import sys as _sys

_sys.modules[__name__] = _impl
