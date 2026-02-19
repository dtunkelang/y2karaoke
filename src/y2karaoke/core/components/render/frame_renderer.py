"""Frame rendering for karaoke videos."""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import math

from ....config import (
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    LINE_SPACING,
    SPLASH_DURATION,
    OUTRO_DELAY,
    LYRICS_ACTIVATION_LEAD,
    FIRST_WORD_HIGHLIGHT_DELAY as _FIRST_WORD_HIGHLIGHT_DELAY,
    Colors,
)
from .backgrounds_static import draw_logo_screen, draw_splash_screen
from .progress import draw_progress_bar
from .lyrics_renderer import get_singer_colors as _get_singer_colors
from .lyric_timeline import (
    check_intro_progress as _check_intro_progress,
    check_mid_song_progress as _check_mid_song_progress,
    get_lines_to_display as _get_lines_to_display,
    check_cue_indicator as _check_cue_indicator,
    carryover_handoff_delay as _carryover_handoff_delay,
)
from .render_text import (
    draw_line_text as _draw_line_text,
    draw_highlight_sweep as _draw_highlight_sweep,
    compute_word_highlight_width as _compute_word_highlight_width_impl,
)
from ...models import Line

# Backward-compatible re-export for tests and existing imports.
get_singer_colors = _get_singer_colors
# Backward-compatible constant export for tests.
FIRST_WORD_HIGHLIGHT_DELAY = _FIRST_WORD_HIGHLIGHT_DELAY


@dataclass(frozen=True)
class RenderPlan:
    """Per-frame render plan derived from timeline and mode decisions."""

    mode: str
    progress: float
    current_line_idx: int
    activation_time: float


def _compute_word_highlight_width(
    line: Line,
    words_with_spaces: list[str],
    word_widths: list[float],
    highlight_time: float,
) -> int:
    """Compatibility wrapper for tests/consumers patching frame_renderer delay constant."""
    import y2karaoke.core.components.render.render_text as render_text_module

    # Keep historical compatibility when tests mutate
    # frame_renderer.FIRST_WORD_HIGHLIGHT_DELAY.
    render_text_module.FIRST_WORD_HIGHLIGHT_DELAY = FIRST_WORD_HIGHLIGHT_DELAY
    return _compute_word_highlight_width_impl(
        line, words_with_spaces, word_widths, highlight_time
    )


def _draw_cue_indicator(
    draw: ImageDraw.ImageDraw, x: int, y: int, time_until_start: float, font_size: int
) -> None:
    """Draw animated cue indicator (pulsing dots) to prepare singer.

    Args:
        draw: PIL ImageDraw object
        x: X position (left side of line)
        y: Y position (vertical center of line)
        time_until_start: Seconds until the line starts
        font_size: Font size for scaling the indicator
    """
    # Three dots that pulse in sequence as countdown
    dot_radius = max(4, font_size // 12)
    dot_spacing = dot_radius * 3
    total_width = dot_spacing * 2 + dot_radius * 2

    # Position dots to the left of the line
    start_x = x - total_width - dot_spacing

    # Calculate which dots to show based on countdown
    # At 3s: 3 dots, at 2s: 2 dots, at 1s: 1 dot (pulsing)
    dots_to_show = min(3, max(1, int(time_until_start) + 1))

    # Pulse animation (sine wave for smooth pulsing)
    pulse = 0.5 + 0.5 * math.sin(time_until_start * math.pi * 3)  # 1.5 Hz pulse

    for i in range(3):
        dot_x = start_x + i * dot_spacing
        dot_y = y

        if i < dots_to_show:
            # Active dot - gold color with pulse on the leading dot
            if i == dots_to_show - 1:
                # Leading dot pulses
                radius = int(dot_radius * (0.8 + 0.4 * pulse))
            else:
                # Other dots are solid
                radius = dot_radius

            color = Colors.CUE_INDICATOR
            draw.ellipse(
                [dot_x - radius, dot_y - radius, dot_x + radius, dot_y + radius],
                fill=color,
            )
        else:
            # Inactive dot - dim outline
            draw.ellipse(
                [
                    dot_x - dot_radius,
                    dot_y - dot_radius,
                    dot_x + dot_radius,
                    dot_y + dot_radius,
                ],
                outline=(100, 100, 100),
                width=1,
            )


def _get_or_build_line_layout(
    line: Line,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    layout_cache: Optional[Dict[int, Tuple[List[str], List[float], float]]],
) -> tuple[list[str], list[float], float]:
    """Return cached line text layout or compute and store it."""
    if layout_cache is not None and id(line) in layout_cache:
        return layout_cache[id(line)]

    words_with_spaces = []
    for i, word in enumerate(line.words):
        words_with_spaces.append(word.text)
        if i < len(line.words) - 1:
            words_with_spaces.append(" ")

    total_width = 0.0
    word_widths = []
    for text in words_with_spaces:
        bbox = font.getbbox(text)
        w = float(bbox[2] - bbox[0])
        word_widths.append(w)
        total_width += w

    if layout_cache is not None:
        layout_cache[id(line)] = (words_with_spaces, word_widths, total_width)
    return words_with_spaces, word_widths, total_width


def _resolve_current_line_idx(lines: list[Line], activation_time: float) -> int:
    """Resolve active line index from activation time and carryover heuristics."""
    current_line_idx = 0
    for i, line in enumerate(lines):
        if line.start_time <= activation_time:
            current_line_idx = i
    if 0 < current_line_idx < len(lines):
        prev_line = lines[current_line_idx - 1]
        curr_line = lines[current_line_idx]
        handoff_delay = _carryover_handoff_delay(prev_line, curr_line)
        if handoff_delay > 0 and activation_time < prev_line.end_time + handoff_delay:
            current_line_idx -= 1
    return current_line_idx


def _compute_line_highlight_width(
    line: Line,
    words_with_spaces: list[str],
    word_widths: list[float],
    total_width: float,
    current_time: float,
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
        return _compute_word_highlight_width(
            line, words_with_spaces, word_widths, highlight_time
        )
    if line_duration > 0:
        line_progress = (highlight_time - line.start_time) / line_duration
        line_progress = max(0.0, min(1.0, line_progress))
    else:
        line_progress = 1.0
    return int(total_width * line_progress)


def _draw_visible_lines(
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
) -> None:
    """Draw all visible lyric lines and their highlight sweeps."""
    for idx, (line, _is_current) in enumerate(lines_to_show):
        y = start_y + idx * LINE_SPACING

        words_with_spaces, word_widths, total_width = _get_or_build_line_layout(
            line, font, layout_cache
        )
        line_x = (video_width - total_width) / 2.0

        if idx == 0 and show_cue:
            font_size = LINE_SPACING * 3 // 4
            _draw_cue_indicator(
                draw, int(line_x), y + font_size // 2, cue_time_until, font_size
            )

        _draw_line_text(
            draw, line, y, line_x, words_with_spaces, word_widths, font, is_duet
        )

        line_duration = line.end_time - line.start_time
        highlight_ready = current_time >= line.start_time
        if line_duration <= 0:
            highlight_ready = current_time + LYRICS_ACTIVATION_LEAD >= line.start_time
        if not (line.words and highlight_ready):
            continue

        highlight_width = _compute_line_highlight_width(
            line,
            words_with_spaces,
            word_widths,
            total_width,
            current_time,
        )
        _draw_highlight_sweep(
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


def _compute_frame_display_state(
    lines: list[Line],
    *,
    current_time: float,
    title: Optional[str],
    artist: Optional[str],
    audio_duration: Optional[float],
) -> RenderPlan:
    """Compute render mode and timeline state for a frame."""
    show_splash = bool(current_time < SPLASH_DURATION and title and artist)
    show_progress_bar, progress = _check_intro_progress(lines, current_time)

    activation_time = current_time + LYRICS_ACTIVATION_LEAD
    current_line_idx = _resolve_current_line_idx(lines, activation_time)

    outro_start = lines[-1].end_time + OUTRO_DELAY if lines else OUTRO_DELAY
    if audio_duration:
        outro_start = max(outro_start, audio_duration - OUTRO_DELAY)
    if lines and current_time >= outro_start:
        return RenderPlan("outro", progress, current_line_idx, activation_time)

    if not show_progress_bar:
        show_progress_bar, progress = _check_mid_song_progress(
            lines, current_line_idx, current_time
        )

    if show_splash:
        return RenderPlan("splash", progress, current_line_idx, activation_time)
    if show_progress_bar:
        return RenderPlan("progress", progress, current_line_idx, activation_time)
    return RenderPlan("lyrics", progress, current_line_idx, activation_time)


def render_frame(  # noqa: C901
    lines: list[Line],
    current_time: float,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    background: np.ndarray,
    title: Optional[str] = None,
    artist: Optional[str] = None,
    is_duet: bool = False,
    width: Optional[int] = None,
    height: Optional[int] = None,
    audio_duration: Optional[float] = None,
    layout_cache: Optional[Dict[int, Tuple[List[str], List[float], float]]] = None,
) -> np.ndarray:
    """Render a single frame at the given time."""
    video_width = width or VIDEO_WIDTH
    video_height = height or VIDEO_HEIGHT
    img = Image.fromarray(background.copy())
    draw = ImageDraw.Draw(img)

    plan = _compute_frame_display_state(
        lines,
        current_time=current_time,
        title=title,
        artist=artist,
        audio_duration=audio_duration,
    )

    if plan.mode == "outro":
        draw_logo_screen(draw, font, video_width, video_height)
        return np.array(img)
    if plan.mode == "splash" and title and artist:
        draw_splash_screen(draw, title, artist, video_width, video_height)
        return np.array(img)
    if plan.mode == "progress":
        draw_progress_bar(draw, plan.progress, video_width, video_height)
        return np.array(img)

    lines_to_show, display_start_idx = _get_lines_to_display(
        lines, plan.current_line_idx, current_time, plan.activation_time
    )
    total_height = len(lines_to_show) * LINE_SPACING
    start_y = (video_height - total_height) // 2

    show_cue, cue_time_until = _check_cue_indicator(
        lines, lines_to_show, display_start_idx, current_time
    )
    _draw_visible_lines(
        draw=draw,
        lines_to_show=lines_to_show,
        start_y=start_y,
        current_time=current_time,
        font=font,
        video_width=video_width,
        is_duet=is_duet,
        show_cue=show_cue,
        cue_time_until=cue_time_until,
        layout_cache=layout_cache,
    )

    return np.array(img)
