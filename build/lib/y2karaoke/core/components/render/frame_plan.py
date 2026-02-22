"""Render-plan helpers for per-frame mode and timeline state."""

from dataclasses import dataclass
from typing import Optional

from ....config import LYRICS_ACTIVATION_LEAD, OUTRO_DELAY, SPLASH_DURATION
from .lyric_timeline import (
    check_intro_progress,
    check_mid_song_progress,
    carryover_handoff_delay,
)
from ...models import Line


@dataclass(frozen=True)
class RenderPlan:
    """Per-frame render plan derived from timeline and mode decisions."""

    mode: str
    progress: float
    current_line_idx: int
    activation_time: float


def resolve_current_line_idx(lines: list[Line], activation_time: float) -> int:
    """Resolve active line index from activation time and carryover heuristics."""
    current_line_idx = 0
    for i, line in enumerate(lines):
        if line.start_time <= activation_time:
            current_line_idx = i
    if 0 < current_line_idx < len(lines):
        prev_line = lines[current_line_idx - 1]
        curr_line = lines[current_line_idx]
        handoff_delay = carryover_handoff_delay(prev_line, curr_line)
        if handoff_delay > 0 and activation_time < prev_line.end_time + handoff_delay:
            current_line_idx -= 1
    return current_line_idx


def compute_frame_display_state(
    lines: list[Line],
    *,
    current_time: float,
    title: Optional[str],
    artist: Optional[str],
    audio_duration: Optional[float],
) -> RenderPlan:
    """Compute render mode and timeline state for a frame."""
    show_splash = bool(current_time < SPLASH_DURATION and title and artist)
    show_progress_bar, progress = check_intro_progress(lines, current_time)

    activation_time = current_time + LYRICS_ACTIVATION_LEAD
    current_line_idx = resolve_current_line_idx(lines, activation_time)

    outro_start = lines[-1].end_time + OUTRO_DELAY if lines else OUTRO_DELAY
    if audio_duration:
        outro_start = max(outro_start, audio_duration - OUTRO_DELAY)
    if lines and current_time >= outro_start:
        return RenderPlan("outro", progress, current_line_idx, activation_time)

    if not show_progress_bar:
        show_progress_bar, progress = check_mid_song_progress(
            lines, current_line_idx, current_time
        )

    if show_splash:
        return RenderPlan("splash", progress, current_line_idx, activation_time)
    if show_progress_bar:
        return RenderPlan("progress", progress, current_line_idx, activation_time)
    return RenderPlan("lyrics", progress, current_line_idx, activation_time)
