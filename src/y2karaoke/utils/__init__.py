"""Utility modules."""

from .logging import setup_logging, get_logger
from .validation import (
    validate_youtube_url,
    validate_key_shift,
    validate_tempo,
    validate_offset,
    validate_output_path,
    sanitize_filename,
)
from .cache import CacheManager

__all__ = [
    "setup_logging",
    "get_logger",
    "validate_youtube_url",
    "validate_key_shift",
    "validate_tempo",
    "validate_offset",
    "validate_output_path",
    "sanitize_filename",
    "CacheManager",
]

try:
    from .fonts import get_font

    __all__.append("get_font")
except ImportError:
    pass
