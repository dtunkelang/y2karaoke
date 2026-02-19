"""Frame rendering for karaoke videos."""

from typing import Optional, Dict, Tuple, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ....config import (
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    LINE_SPACING,
    FIRST_WORD_HIGHLIGHT_DELAY as _FIRST_WORD_HIGHLIGHT_DELAY,
)
from .backgrounds_static import draw_logo_screen, draw_splash_screen
from .cue_indicator import draw_cue_indicator as _draw_cue_indicator_impl
from .frame_plan import (
    RenderPlan as _RenderPlan,
    resolve_current_line_idx as _resolve_current_line_idx_impl,
    compute_frame_display_state as _compute_frame_display_state_impl,
)
from .line_pass import (
    compute_line_highlight_width as _compute_line_highlight_width_impl,
    draw_visible_lines as _draw_visible_lines_impl,
)
from .progress import draw_progress_bar
from .lyrics_renderer import get_singer_colors as _get_singer_colors
from .layout import get_or_build_line_layout as _get_or_build_line_layout_impl
from .lyric_timeline import (
    check_intro_progress as _check_intro_progress_impl,
    check_mid_song_progress as _check_mid_song_progress_impl,
    get_lines_to_display as _get_lines_to_display,
    check_cue_indicator as _check_cue_indicator,
    carryover_handoff_delay as _carryover_handoff_delay_impl,
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

# Backward-compatible type export for tests and imports.
RenderPlan = _RenderPlan


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
    """Compatibility wrapper around cue-indicator rendering primitive."""
    _draw_cue_indicator_impl(draw, x, y, time_until_start, font_size)


def _get_or_build_line_layout(
    line: Line,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    layout_cache: Optional[Dict[int, Tuple[List[str], List[float], float]]],
) -> tuple[list[str], list[float], float]:
    """Compatibility wrapper around line layout helper primitive."""
    return _get_or_build_line_layout_impl(line, font, layout_cache)


def _check_intro_progress(lines: list[Line], current_time: float) -> tuple[bool, float]:
    """Compatibility wrapper around intro progress timeline helper."""
    return _check_intro_progress_impl(lines, current_time)


def _check_mid_song_progress(
    lines: list[Line], current_line_idx: int, current_time: float
) -> tuple[bool, float]:
    """Compatibility wrapper around mid-song progress timeline helper."""
    return _check_mid_song_progress_impl(lines, current_line_idx, current_time)


def _carryover_handoff_delay(prev_line: Line, next_line: Line) -> float:
    """Compatibility wrapper around carryover handoff helper."""
    return _carryover_handoff_delay_impl(prev_line, next_line)


def _resolve_current_line_idx(lines: list[Line], activation_time: float) -> int:
    """Compatibility wrapper around frame-plan line index resolution."""
    return _resolve_current_line_idx_impl(lines, activation_time)


def _compute_line_highlight_width(
    line: Line,
    words_with_spaces: list[str],
    word_widths: list[float],
    total_width: float,
    current_time: float,
) -> int:
    """Compatibility wrapper around line-pass highlight-width helper."""
    return _compute_line_highlight_width_impl(
        line,
        words_with_spaces,
        word_widths,
        total_width,
        current_time,
        compute_word_highlight_width_fn=_compute_word_highlight_width,
    )


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
    """Compatibility wrapper around line-pass drawing helper."""
    _draw_visible_lines_impl(
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
        get_or_build_line_layout_fn=_get_or_build_line_layout,
        draw_cue_indicator_fn=_draw_cue_indicator,
        draw_line_text_fn=_draw_line_text,
        draw_highlight_sweep_fn=_draw_highlight_sweep,
        compute_word_highlight_width_fn=_compute_word_highlight_width,
    )


def _compute_frame_display_state(
    lines: list[Line],
    *,
    current_time: float,
    title: Optional[str],
    artist: Optional[str],
    audio_duration: Optional[float],
) -> RenderPlan:
    """Compatibility wrapper around frame-plan display-state computation."""
    return _compute_frame_display_state_impl(
        lines,
        current_time=current_time,
        title=title,
        artist=artist,
        audio_duration=audio_duration,
    )


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
