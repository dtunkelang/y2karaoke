"""Singer styling logic for karaoke videos."""

import os
from typing import Tuple

from ....config import Colors


def get_singer_colors(singer: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Get text and highlight colors for a singer."""
    color_mode = os.getenv("Y2KARAOKE_SINGER_COLOR_MODE", "auto").strip().lower()
    if color_mode in {"single", "mono", "one", "off"}:
        return (Colors.TEXT, Colors.HIGHLIGHT)

    if singer == "singer1":
        return (Colors.SINGER1, Colors.SINGER1_HIGHLIGHT)
    elif singer == "singer2":
        return (Colors.SINGER2, Colors.SINGER2_HIGHLIGHT)
    elif singer == "both":
        return (Colors.BOTH, Colors.BOTH_HIGHLIGHT)
    else:
        # Default colors (gold highlight, white text)
        return (Colors.TEXT, Colors.HIGHLIGHT)
