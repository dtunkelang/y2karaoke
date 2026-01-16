"""Font utilities for video rendering."""

import os
from pathlib import Path
from typing import Optional

from PIL import ImageFont

from ..config import FONT_SIZE
from ..utils.logging import get_logger

logger = get_logger(__name__)

def get_font(size: int = FONT_SIZE) -> ImageFont.FreeTypeFont:
    """Get font for rendering text."""
    
    # Font search paths (macOS, Linux, Windows)
    font_paths = [
        # macOS
        "/System/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    
    # Try to find a suitable font
    for font_path in font_paths:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
    
    # Fallback to default font
    try:
        return ImageFont.load_default()
    except Exception:
        logger.warning("Could not load any font, using PIL default")
        return ImageFont.load_default()
