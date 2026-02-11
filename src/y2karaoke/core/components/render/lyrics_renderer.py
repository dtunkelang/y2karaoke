"""Lyrics rendering utilities for karaoke videos."""

import os

from ....config import Colors


def get_singer_colors(
    singer: str, is_highlighted: bool
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Get text and highlight colors for a singer."""
    _ = is_highlighted
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
