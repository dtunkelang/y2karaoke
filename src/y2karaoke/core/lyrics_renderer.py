"""Lyrics rendering utilities for karaoke videos."""

from ..config import Colors


def get_singer_colors(
    singer: str, is_highlighted: bool
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Get text and highlight colors for a singer."""
    if singer == "singer1":
        return (Colors.SINGER1, Colors.SINGER1_HIGHLIGHT)
    elif singer == "singer2":
        return (Colors.SINGER2, Colors.SINGER2_HIGHLIGHT)
    elif singer == "both":
        return (Colors.BOTH, Colors.BOTH_HIGHLIGHT)
    else:
        # Default colors (gold highlight, white text)
        return (Colors.TEXT, Colors.HIGHLIGHT)
