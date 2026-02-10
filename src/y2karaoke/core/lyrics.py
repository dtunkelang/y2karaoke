"""Compatibility facade for lyrics processing APIs."""

from .components.lyrics import api as _api
from .components.lyrics.api import *  # noqa: F401,F403

__all__ = _api.__all__
