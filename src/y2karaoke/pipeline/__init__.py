"""Pipeline subsystem facades.

These packages expose stable orchestration boundaries while core modules
continue to host the implementation details.
"""

from . import alignment, audio, identify, lyrics, render

__all__ = ["alignment", "audio", "identify", "lyrics", "render"]
