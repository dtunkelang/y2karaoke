"""Compatibility module alias for lyrics rendering helpers."""

from .components.render import lyrics_renderer as _impl
import sys as _sys

_sys.modules[__name__] = _impl
