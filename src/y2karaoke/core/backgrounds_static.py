"""Compatibility module alias for static background rendering."""

from .components.render import backgrounds_static as _impl
import sys as _sys

_sys.modules[__name__] = _impl
