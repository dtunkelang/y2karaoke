"""Visible-line rendering pass helpers."""

from typing import Callable, Dict, List, Optional, Tuple

from PIL import ImageDraw, ImageFont

from ....config import LINE_SPACING, LYRICS_ACTIVATION_LEAD
from ...models import Line


def compute_line_highlight_width(
    line: Line,
    words_with_spaces: list[str],
    word_widths: list[float],
    total_width: float,
    current_time: float,
    *,
    compute_word_highlight_width_fn: Callable[
        [Line, list[str], list[float], float], int
    ],
) -> int:
    """Compute highlight width for the current line at frame time."""
    line_duration = line.end_time - line.start_time
    highlight_time = current_time
    if highlight_time >= line.end_time:
        highlight_time = line.end_time
    if line_duration <= 0:
        highlight_time = line.end_time
    if line.words and all(
        w.start_time is not None and w.end_time is not None for w in line.words
    ):
        return compute_word_highlight_width_fn(
            line, words_with_spaces, word_widths, highlight_time
        )
    if line_duration > 0:
        line_progress = (highlight_time - line.start_time) / line_duration
        line_progress = max(0.0, min(1.0, line_progress))
    else:
        line_progress = 1.0
    return int(total_width * line_progress)


def draw_visible_lines(
    *,
    draw: ImageDraw.ImageDraw,
    lines_to_show: list[tuple[Line, bool]],
    start_y: int,
    current_time: float,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    video_width: int,
    is_duet: bool,
    show_cue: bool,
    cue_time_until: float,
    layout_cache: Optional[Dict[int, Tuple[List[str], List[float], float]]],
    get_or_build_line_layout_fn: Callable[
        [
            Line,
            ImageFont.ImageFont | ImageFont.FreeTypeFont,
            Optional[Dict[int, Tuple[List[str], List[float], float]]],
        ],
        tuple[list[str], list[float], float],
    ],
    draw_cue_indicator_fn: Callable[[ImageDraw.ImageDraw, int, int, float, int], None],
    draw_line_text_fn: Callable[
        [
            ImageDraw.ImageDraw,
            Line,
            int,
            float,
            list[str],
            list[float],
            ImageFont.ImageFont | ImageFont.FreeTypeFont,
            bool,
        ],
        None,
    ],
    draw_highlight_sweep_fn: Callable[
        [
            ImageDraw.ImageDraw,
            Line,
            int,
            float,
            float,
            list[str],
            list[float],
            ImageFont.ImageFont | ImageFont.FreeTypeFont,
            int,
            bool,
        ],
        None,
    ],
    compute_word_highlight_width_fn: Callable[
        [Line, list[str], list[float], float], int
    ],
) -> None:
    """Draw all visible lyric lines and their highlight sweeps."""
    for idx, (line, _is_current) in enumerate(lines_to_show):
        y = start_y + idx * LINE_SPACING
        words_with_spaces, word_widths, total_width = get_or_build_line_layout_fn(
            line, font, layout_cache
        )
        line_x = (video_width - total_width) / 2.0

        if idx == 0 and show_cue:
            font_size = LINE_SPACING * 3 // 4
            draw_cue_indicator_fn(
                draw, int(line_x), y + font_size // 2, cue_time_until, font_size
            )

        draw_line_text_fn(
            draw, line, y, line_x, words_with_spaces, word_widths, font, is_duet
        )

        line_duration = line.end_time - line.start_time
        highlight_ready = current_time >= line.start_time
        if line_duration <= 0:
            highlight_ready = current_time + LYRICS_ACTIVATION_LEAD >= line.start_time
        if not (line.words and highlight_ready):
            continue

        highlight_width = compute_line_highlight_width(
            line,
            words_with_spaces,
            word_widths,
            total_width,
            current_time,
            compute_word_highlight_width_fn=compute_word_highlight_width_fn,
        )
        draw_highlight_sweep_fn(
            draw,
            line,
            y,
            line_x,
            total_width,
            words_with_spaces,
            word_widths,
            font,
            highlight_width,
            is_duet,
        )
