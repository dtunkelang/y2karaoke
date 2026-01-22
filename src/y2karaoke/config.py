"""Configuration settings for Y2Karaoke."""

import os
from pathlib import Path
from typing import Tuple, Optional

from .exceptions import ConfigError

# Directories
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "karaoke"
DEFAULT_OUTPUT_DIR = Path.cwd()

# Video settings (can be overridden via environment variables)
VIDEO_WIDTH = int(os.getenv("Y2KARAOKE_VIDEO_WIDTH", "1920"))
VIDEO_HEIGHT = int(os.getenv("Y2KARAOKE_VIDEO_HEIGHT", "1080"))
FPS = int(os.getenv("Y2KARAOKE_FPS", "30"))

# Resolution presets
RESOLUTION_PRESETS = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "4k": (3840, 2160),
}

# Colors (RGB tuples)
class Colors:
    BG_TOP = (20, 20, 40)
    BG_BOTTOM = (40, 20, 60)
    TEXT = (255, 255, 255)
    HIGHLIGHT = (255, 215, 0)
    SUNG = (180, 180, 180)
    PROGRESS_BG = (60, 60, 80)
    PROGRESS_FG = (255, 215, 0)
    
    # Duet colors
    SINGER1 = (100, 180, 255)
    SINGER1_HIGHLIGHT = (50, 150, 255)
    SINGER2 = (255, 150, 200)
    SINGER2_HIGHLIGHT = (255, 100, 180)
    BOTH = (200, 150, 255)
    BOTH_HIGHLIGHT = (180, 100, 255)

# Font settings (can be overridden via environment variables)
FONT_SIZE = int(os.getenv("Y2KARAOKE_FONT_SIZE", "72"))
LINE_SPACING = int(os.getenv("Y2KARAOKE_LINE_SPACING", "100"))

# Audio processing
AUDIO_SAMPLE_RATE = 44100
KEY_SHIFT_RANGE = (-12, 12)
TEMPO_RANGE = (0.1, 3.0)

# Video timing
SPLASH_DURATION = 4.0
INSTRUMENTAL_BREAK_THRESHOLD = 8.0  # Show instrumental break screen for gaps >= 8 seconds
LYRICS_LEAD_TIME = 1.0
HIGHLIGHT_LEAD_TIME = 0.15  # Seconds to advance highlight ahead of audio

# Line splitting
MAX_LINE_WIDTH_RATIO = 0.75  # 75% of screen width

def validate_config() -> None:
    """Validate configuration values."""
    if not (0.1 <= TEMPO_RANGE[0] <= TEMPO_RANGE[1] <= 3.0):
        raise ConfigError("Invalid tempo range")
    
    if not (-12 <= KEY_SHIFT_RANGE[0] <= KEY_SHIFT_RANGE[1] <= 12):
        raise ConfigError("Invalid key shift range")
    
    if VIDEO_WIDTH <= 0 or VIDEO_HEIGHT <= 0:
        raise ConfigError("Invalid video dimensions")
    
    if FPS <= 0:
        raise ConfigError("Invalid FPS value")

def get_cache_dir() -> Path:
    """Get cache directory from environment or default."""
    cache_dir = os.getenv("Y2KARAOKE_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir)
    return DEFAULT_CACHE_DIR

# Validate config on import
validate_config()

# Background processing
DARKEN_FACTOR = 0.4
BLUR_RADIUS = 3

# Cache settings
MAX_CACHE_SIZE_GB = 10
CACHE_CLEANUP_THRESHOLD = 0.8

def get_credentials_dir() -> Path:
    """Get credentials directory."""
    return get_cache_dir()


def parse_resolution(resolution_str: str) -> Tuple[int, int]:
    """
    Parse resolution string into (width, height) tuple.

    Args:
        resolution_str: Resolution as "WIDTHxHEIGHT" (e.g., "1920x1080")
                       or preset name (e.g., "720p", "1080p", "4k")

    Returns:
        Tuple of (width, height)

    Raises:
        ValueError: If resolution string is invalid
    """
    resolution_str = resolution_str.lower().strip()

    # Check if it's a preset
    if resolution_str in RESOLUTION_PRESETS:
        return RESOLUTION_PRESETS[resolution_str]

    # Try to parse as WIDTHxHEIGHT
    if "x" in resolution_str:
        try:
            parts = resolution_str.split("x")
            width = int(parts[0])
            height = int(parts[1])
            if width > 0 and height > 0:
                return (width, height)
        except (ValueError, IndexError):
            pass

    raise ValueError(
        f"Invalid resolution: {resolution_str}. "
        f"Use format 'WIDTHxHEIGHT' (e.g., '1920x1080') or "
        f"preset name: {', '.join(RESOLUTION_PRESETS.keys())}"
    )
