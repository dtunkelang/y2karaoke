"""Configuration settings for Y2Karaoke."""

import os
from pathlib import Path
from typing import Tuple

# Directories
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "karaoke"
DEFAULT_OUTPUT_DIR = Path.cwd()

# Video settings
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
FPS = 30

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

# Font settings
FONT_SIZE = 72
LINE_SPACING = 100

# Audio processing
AUDIO_SAMPLE_RATE = 44100
KEY_SHIFT_RANGE = (-12, 12)
TEMPO_RANGE = (0.1, 3.0)

# Timing settings
INSTRUMENTAL_BREAK_THRESHOLD = 5.0
LYRICS_LEAD_TIME = 1.0
SPLASH_DURATION = 4.0

# Background processing
DARKEN_FACTOR = 0.4
BLUR_RADIUS = 3

# Cache settings
MAX_CACHE_SIZE_GB = 10
CACHE_CLEANUP_THRESHOLD = 0.8

# API settings
YOUTUBE_API_SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

def get_cache_dir() -> Path:
    """Get cache directory from environment or default."""
    return Path(os.getenv("Y2KARAOKE_CACHE_DIR", DEFAULT_CACHE_DIR))

def get_credentials_dir() -> Path:
    """Get credentials directory."""
    return get_cache_dir()
