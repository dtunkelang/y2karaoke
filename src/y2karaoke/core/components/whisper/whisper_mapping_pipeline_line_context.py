"""Line context and drift-clamp helpers for Whisper mapping pipeline."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from ... import models
from . import whisper_utils
from .whisper_dtw import _LineMappingContext
from .whisper_mapping_helpers import _choose_segment_for_line, _find_segment_for_time


def _fallback_unmatched_line_duration(line: "models.Line") -> float:
    """Estimate a reasonable fallback duration when no Whisper words match a line."""
    original_duration = line.end_time - line.start_time
    if original_duration <= 0:
        return max(len(line.words) * 0.6, 0.25)

    if not line.words:
        return max(original_duration, 0.2)

    if len(line.words) == 1:
        min_duration = 0.22
    else:
        min_duration = max(0.35, len(line.words) * 0.16)
    return max(original_duration, min_duration)


def _prepare_line_context(
    ctx: _LineMappingContext,
    line: "models.Line",
    *,
    max_anchor_drift_from_lrc: float,
) -> Tuple[Optional[int], float, float]:
    """Determine segment, anchor time, and shift for a line."""
    line_segment = _choose_segment_for_line(
        line,
        ctx.segments,
        ctx.current_segment,
        min_start=ctx.last_line_start,
        excluded_segments=ctx.used_segments,
    )
    if line_segment is None:
        line_segment = _find_segment_for_time(
            line.start_time,
            ctx.segments,
            ctx.current_segment,
            excluded_segments=ctx.used_segments,
        )
    if (
        line_segment is not None
        and ctx.segments
        and whisper_utils._segment_start(ctx.segments[line_segment])
        < ctx.last_line_start
    ):
        line_segment = None
    prior_anchor = max(ctx.last_line_start, ctx.prev_line_end)
    if prior_anchor == float("-inf"):
        line_anchor_time = line.start_time
    else:
        line_anchor_time = max(
            line.start_time,
            min(prior_anchor, line.start_time + max_anchor_drift_from_lrc),
        )
    line_shift = line_anchor_time - line.start_time
    return line_segment, line_anchor_time, line_shift


def _should_override_line_segment(
    *,
    current_segment: Optional[int],
    override_segment: int,
    override_hits: int,
    line_word_count: int,
    line_anchor_time: float,
    segments: Sequence[Any],
    max_local_jump_seconds: float = 8.0,
    max_strong_jump_seconds: float = 18.0,
    max_anchor_jump_seconds: float = 14.0,
    max_anchor_strong_jump_seconds: float = 20.0,
) -> bool:
    """Decide whether assignment-based segment override is trustworthy."""
    strong_hits = override_hits >= max(2, int(0.6 * max(1, line_word_count)))

    if current_segment is None:
        if not segments:
            return True
        override_start = whisper_utils._segment_start(segments[override_segment])
        anchor_jump = abs(override_start - line_anchor_time)
        if anchor_jump <= max_anchor_jump_seconds:
            return True
        return strong_hits and anchor_jump <= max_anchor_strong_jump_seconds

    if not segments:
        return abs(override_segment - current_segment) <= 2

    current_start = whisper_utils._segment_start(segments[current_segment])
    override_start = whisper_utils._segment_start(segments[override_segment])
    jump_seconds = abs(override_start - current_start)
    anchor_jump = abs(override_start - line_anchor_time)

    if (
        jump_seconds <= max_local_jump_seconds
        and anchor_jump <= max_anchor_jump_seconds
    ):
        return True

    return (
        strong_hits
        and jump_seconds <= max_strong_jump_seconds
        and anchor_jump <= max_anchor_strong_jump_seconds
    )


def _clamp_match_window_to_anchor(
    actual_start: float,
    actual_end: float,
    line_anchor_time: float,
    *,
    max_forward: float,
    max_backward: float,
) -> Tuple[float, float]:
    """Keep mapped line windows from drifting implausibly far from anchor time."""
    if actual_end <= actual_start:
        actual_end = actual_start + 0.2
    if line_anchor_time == float("-inf"):
        return actual_start, actual_end

    min_start = line_anchor_time - max_backward
    max_start = line_anchor_time + max_forward
    clamped_start = min(max(actual_start, min_start), max_start)
    if clamped_start == actual_start:
        return actual_start, actual_end

    shift = clamped_start - actual_start
    clamped_end = max(clamped_start + 0.2, actual_end + shift)
    return clamped_start, clamped_end


def _clamp_line_shift_vs_original(
    mapped_line: "models.Line",
    original_line: "models.Line",
    *,
    max_forward: float,
    max_backward: float,
) -> "models.Line":
    """Clamp per-line timing drift relative to original LRC timing."""
    if not mapped_line.words or not original_line.words:
        return mapped_line

    delta = mapped_line.start_time - original_line.start_time
    if -max_backward <= delta <= max_forward:
        return mapped_line

    clamped_delta = min(max(delta, -max_backward), max_forward)
    shift = clamped_delta - delta
    shifted_words = [
        models.Word(
            text=w.text,
            start_time=w.start_time + shift,
            end_time=w.end_time + shift,
            singer=w.singer,
        )
        for w in mapped_line.words
    ]
    return models.Line(words=shifted_words, singer=mapped_line.singer)


def _clamp_line_duration_vs_original(
    mapped_line: "models.Line",
    original_line: "models.Line",
    next_original_start: Optional[float],
    *,
    max_scale: float,
    slack_seconds: float = 0.9,
) -> "models.Line":
    """Prevent mapped line durations from bleeding far past expected LRC span."""
    if not mapped_line.words or not original_line.words:
        return mapped_line

    mapped_duration = mapped_line.end_time - mapped_line.start_time
    original_duration = max(0.2, original_line.end_time - original_line.start_time)
    cap = max(original_duration * max_scale, original_duration + 0.8)

    if next_original_start is not None:
        expected_span = max(0.2, next_original_start - original_line.start_time)
        cap = min(cap, expected_span + slack_seconds)

    if mapped_duration <= cap:
        return mapped_line

    scale = cap / max(mapped_duration, 0.01)
    start = mapped_line.start_time
    new_words = []
    for w in mapped_line.words:
        ws = start + (w.start_time - start) * scale
        we = start + (w.end_time - start) * scale
        if we < ws:
            we = ws
        new_words.append(
            models.Word(
                text=w.text,
                start_time=ws,
                end_time=we,
                singer=w.singer,
            )
        )
    return models.Line(words=new_words, singer=mapped_line.singer)
