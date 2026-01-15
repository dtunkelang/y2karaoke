"""Font utilities for cross-platform font loading."""

import os
import sys
from functools import lru_cache
from typing import Optional

from PIL import ImageFont

from ..config import FONT_SIZE


# Platform-specific font paths in order of preference
FONT_PATHS = {
    'darwin': [  # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Arial.ttf",
    ],
    'linux': [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ],
    'win32': [  # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/verdana.ttf",
    ],
}


def _get_platform_fonts() -> list[str]:
    """Get font paths for the current platform."""
    platform = sys.platform
    if platform.startswith('linux'):
        platform = 'linux'
    return FONT_PATHS.get(platform, FONT_PATHS['linux'])


@lru_cache(maxsize=8)
def get_font(size: int = FONT_SIZE) -> ImageFont.FreeTypeFont:
    """
    Get a suitable font for rendering, with cross-platform support.

    Args:
        size: Font size in points (default from config)

    Returns:
        PIL ImageFont object
    """
    font_paths = _get_platform_fonts()

    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue

    # Try all platforms as fallback
    for platform_fonts in FONT_PATHS.values():
        for path in platform_fonts:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue

    # Last resort: PIL default font
    return ImageFont.load_default()


def get_font_path() -> Optional[str]:
    """
    Get the path to the font that will be used for rendering.

    Returns:
        Path to font file, or None if using default
    """
    font_paths = _get_platform_fonts()

    for path in font_paths:
        if os.path.exists(path):
            return path

    # Try all platforms as fallback
    for platform_fonts in FONT_PATHS.values():
        for path in platform_fonts:
            if os.path.exists(path):
                return path

    return None
