"""Lyric timeline helpers: decide which lines should be visible and when."""

from ....config import (
    CARRYOVER_HANDOFF_DELAY_MAX,
    CUE_INDICATOR_DURATION,
    CUE_INDICATOR_MIN_GAP,
    INSTRUMENTAL_BREAK_THRESHOLD,
    LYRICS_LEAD_TIME,
    SPLASH_DURATION,
)
from ...models import Line


def check_intro_progress(lines: list[Line], current_time: float) -> tuple[bool, float]:
    """Check if intro progress bar should be shown before first lyric line."""
    if not lines or current_time >= lines[0].start_time:
        return False, 0.0

    first_line = lines[0]
    time_until_first = first_line.start_time - current_time

    if (
        first_line.start_time >= INSTRUMENTAL_BREAK_THRESHOLD
        and time_until_first > LYRICS_LEAD_TIME
    ):
        bar_start = min(SPLASH_DURATION, first_line.start_time - LYRICS_LEAD_TIME)
        break_end = first_line.start_time - LYRICS_LEAD_TIME
        elapsed = current_time - bar_start
        bar_duration = break_end - bar_start
        progress = elapsed / bar_duration if bar_duration > 0 else 1.0
        return True, progress

    return False, 0.0


def check_mid_song_progress(
    lines: list[Line], current_line_idx: int, current_time: float
) -> tuple[bool, float]:
    """Check if progress bar should be shown during mid-song instrumental break."""
    if current_line_idx >= len(lines):
        return False, 0.0

    current_line = lines[current_line_idx]
    next_line_idx = current_line_idx + 1

    if next_line_idx >= len(lines) or current_time < current_line.end_time:
        return False, 0.0

    next_line = lines[next_line_idx]
    gap = next_line.start_time - current_line.end_time

    if gap >= INSTRUMENTAL_BREAK_THRESHOLD:
        time_until_next = next_line.start_time - current_time
        if time_until_next > LYRICS_LEAD_TIME:
            break_start = current_line.end_time
            break_end = next_line.start_time - LYRICS_LEAD_TIME
            break_duration = break_end - break_start
            elapsed = current_time - break_start
            progress = elapsed / break_duration if break_duration > 0 else 1.0
            return True, progress

    return False, 0.0


def get_lines_to_display(
    lines: list[Line],
    current_line_idx: int,
    current_time: float,
    activation_time: float,
) -> tuple[list[tuple[Line, bool]], int]:
    """Determine visible line window and current active line."""
    display_start_idx = (current_line_idx // 3) * 3
    break_floor_idx = None

    for idx in range(current_line_idx, 0, -1):
        prev_line = lines[idx - 1]
        curr_line = lines[idx]
        gap = curr_line.start_time - prev_line.end_time
        if gap >= INSTRUMENTAL_BREAK_THRESHOLD:
            break_floor_idx = idx
            break

    if break_floor_idx is not None:
        offset = (current_line_idx - break_floor_idx) // 3
        display_start_idx = break_floor_idx + offset * 3

    next_line_idx = current_line_idx + 1
    if next_line_idx < len(lines):
        curr_line = lines[current_line_idx]
        next_line = lines[next_line_idx]
        gap = next_line.start_time - curr_line.end_time
        if gap >= INSTRUMENTAL_BREAK_THRESHOLD and current_time >= curr_line.end_time:
            display_start_idx = next_line_idx
            current_line_idx = next_line_idx

    lines_to_show: list[tuple[Line, bool]] = []
    for i in range(4):
        line_idx = display_start_idx + i
        if line_idx >= len(lines):
            break
        if line_idx > 0:
            prev_line = lines[line_idx - 1]
            this_line = lines[line_idx]
            gap = this_line.start_time - prev_line.end_time
            if (
                gap >= INSTRUMENTAL_BREAK_THRESHOLD
                and current_time < this_line.start_time - LYRICS_LEAD_TIME
            ):
                break
        is_current = (
            line_idx == current_line_idx
            and activation_time >= lines[line_idx].start_time
        )
        lines_to_show.append((lines[line_idx], is_current))

    return lines_to_show, display_start_idx


def check_cue_indicator(
    lines: list[Line],
    lines_to_show: list[tuple[Line, bool]],
    display_start_idx: int,
    current_time: float,
) -> tuple[bool, float]:
    """Check if cue indicator should be shown for the first visible line."""
    if not lines_to_show:
        return False, 0.0

    first_line = lines_to_show[0][0]
    time_until_first = first_line.start_time - current_time

    if display_start_idx == 0:
        gap_before = first_line.start_time
    else:
        prev_line = lines[display_start_idx - 1]
        gap_before = first_line.start_time - prev_line.end_time

    if (
        gap_before >= CUE_INDICATOR_MIN_GAP
        and 0 < time_until_first <= CUE_INDICATOR_DURATION
    ):
        return True, time_until_first

    return False, 0.0


def carryover_handoff_delay(prev_line: Line, next_line: Line) -> float:
    """Compute extra highlight handoff delay for line-continuation phrases."""
    if not prev_line.words or not next_line.words:
        return 0.0
    if "," not in prev_line.text:
        return 0.0
    if len(prev_line.words) < 4:
        return 0.0

    gap = next_line.start_time - prev_line.end_time
    if gap >= 0.8:
        return 0.0

    def _norm_tokens(text: str) -> list[str]:
        return ["".join(ch for ch in t.lower() if ch.isalpha()) for t in text.split()]

    prev_tokens = [t for t in _norm_tokens(prev_line.text) if t]
    next_tokens = [t for t in _norm_tokens(next_line.text) if t]
    overlap = 0
    max_n = min(len(prev_tokens), len(next_tokens), 3)
    for n in range(max_n, 0, -1):
        if prev_tokens[-n:] == next_tokens[:n]:
            overlap = n
            break

    boost = max(0.0, 0.8 - max(gap, 0.0)) * 0.3
    overlap_boost = 0.5 * overlap
    return min(CARRYOVER_HANDOFF_DELAY_MAX, 0.2 + boost + overlap_boost)
