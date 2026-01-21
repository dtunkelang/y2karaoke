"""Lyrics rendering utilities for karaoke videos."""

from typing import Optional, Any
from PIL import ImageDraw

from ..config import LINE_SPACING, Colors


def get_singer_colors(singer: str, is_highlighted: bool) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
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


def draw_lyrics_frame(
    draw: ImageDraw.Draw,
    t: float,
    lines: list,
    font,
    height: int,
    is_duet: bool = False,
    song_metadata: Optional[Any] = None
):
    """
    Draw current and upcoming lyrics on the frame.

    Args:
        draw: PIL ImageDraw object
        t: current timestamp in seconds
        lines: list of Line objects with word-level timing
        font: PIL font
        height: video frame height (for vertical centering)
        is_duet: whether this song has multiple singers
        song_metadata: optional metadata for duet detection
    """
    # Find current and next lines
    current_line_idx = None
    next_line_idx = None

    for i, line in enumerate(lines):
        if line.start_time <= t <= line.end_time:
            current_line_idx = i
            break
        elif line.start_time > t:
            next_line_idx = i
            break

    # Draw current line
    if current_line_idx is not None:
        line = lines[current_line_idx]
        y_pos = height // 2 - LINE_SPACING // 2
        _draw_line(draw, line, t, y_pos, font, is_current=True, is_duet=is_duet)

    # Draw next line
    if next_line_idx is not None:
        line = lines[next_line_idx]
        y_pos = height // 2 + LINE_SPACING // 2
        _draw_line(draw, line, t, y_pos, font, is_current=False, is_duet=is_duet)


def _draw_line(
    draw: ImageDraw.Draw,
    line,
    t: float,
    y_pos: int,
    font,
    is_current: bool,
    is_duet: bool = False
):
    """Draw a single line of lyrics centered horizontally."""
    # Compute total width
    total_width = 0
    word_widths = []
    for word in line.words:
        bbox = draw.textbbox((0, 0), word.text + " ", font=font)
        width = bbox[2] - bbox[0]
        word_widths.append(width)
        total_width += width

    x_pos = (draw.im.size[0] - total_width) // 2  # center

    for i, word in enumerate(line.words):
        # Determine word color
        if is_duet and getattr(word, "singer", None):
            text_color, highlight_color = get_singer_colors(word.singer, False)
            color = highlight_color if is_current and t >= word.start_time else text_color
        else:
            if is_current:
                color = Colors.HIGHLIGHT if t >= word.start_time else Colors.TEXT
            else:
                color = Colors.TEXT

        draw.text((x_pos, y_pos), word.text, font=font, fill=color)
        x_pos += word_widths[i]
