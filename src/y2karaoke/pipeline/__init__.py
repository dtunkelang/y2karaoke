"""Pipeline subsystem facades.

These packages expose stable orchestration boundaries while core modules
continue to host the implementation details.
"""

from . import lyrics, alignment

__all__ = ["lyrics", "alignment"]
