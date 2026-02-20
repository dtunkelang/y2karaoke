"""Mapped-line assembly helpers for Whisper mapping pipeline."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from ... import models
from .whisper_dtw import _LineMappingContext


def _assemble_mapped_line(
    ctx: _LineMappingContext,
    line_idx: int,
    line: "models.Line",
    line_matches: List[Tuple[int, Tuple[float, float]]],
    line_match_intervals: Dict[int, Tuple[float, float]],
    line_anchor_time: float,
    line_segment: Optional[int],
    line_last_idx_ref: List[Optional[int]],
    next_original_start: Optional[float],
    *,
    clamp_match_window_to_anchor_fn: Callable[..., Tuple[float, float]],
    fallback_unmatched_line_duration_fn: Callable[["models.Line"], float],
    redistribute_word_timings_to_line_fn: Callable[..., "models.Line"],
    clamp_line_shift_vs_original_fn: Callable[
        ["models.Line", "models.Line"], "models.Line"
    ],
    clamp_line_duration_vs_original_fn: Callable[
        ["models.Line", "models.Line", Optional[float]], "models.Line"
    ],
    logger,
) -> "models.Line":
    """Build a mapped line from match intervals and update mapping state."""
    from ...models import Line as LineModel

    mapped_words: List[models.Word] = []
    for word_idx, word in enumerate(line.words):
        interval = line_match_intervals.get(word_idx)
        if interval:
            start, end = interval
        else:
            start, end = word.start_time, word.end_time
        mapped_words.append(
            models.Word(
                text=word.text,
                start_time=start,
                end_time=end,
                singer=word.singer,
            )
        )

    mapped_line = LineModel(words=mapped_words, singer=line.singer)
    if line_matches:
        actual_start = min(iv[0] for _, iv in line_matches)
        actual_end = max(iv[1] for _, iv in line_matches)
        actual_start, actual_end = clamp_match_window_to_anchor_fn(
            actual_start,
            actual_end,
            line_anchor_time,
        )
    else:
        original_duration = fallback_unmatched_line_duration_fn(line)
        actual_start = line_anchor_time
        actual_end = actual_start + original_duration
    target_duration = max(actual_end - actual_start, 0.0)
    mapped_line = redistribute_word_timings_to_line_fn(
        mapped_line,
        line_matches,
        target_duration=target_duration,
        min_word_duration=0.05,
        line_start=actual_start,
    )
    mapped_line = clamp_line_shift_vs_original_fn(mapped_line, line)
    mapped_line = clamp_line_duration_vs_original_fn(
        mapped_line, line, next_original_start
    )
    logger.debug(
        "Mapped line %d start=%.2f end=%.2f matches=%d",
        line_idx + 1,
        mapped_line.start_time,
        mapped_line.end_time,
        len(line_matches),
    )
    if mapped_line.words:
        ctx.last_line_start = mapped_line.start_time
        ctx.prev_line_end = mapped_line.end_time
    if line_segment is not None:
        ctx.used_segments.add(line_segment)
    if line_last_idx_ref[0] is not None:
        ctx.next_word_idx_start = line_last_idx_ref[0] + 1
    return mapped_line
