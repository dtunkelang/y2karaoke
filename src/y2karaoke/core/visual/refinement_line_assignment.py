"""Line-level timing-to-word distribution helper."""

from __future__ import annotations

import math
from typing import List, Optional

from ..models import TargetLine


def assign_line_level_word_timings(
    ln: TargetLine,
    line_start: Optional[float],
    line_end: Optional[float],
    line_confidence: float,
) -> None:
    n_words = len(ln.words)
    if n_words == 0:
        ln.word_starts = []
        ln.word_ends = []
        ln.word_confidences = []
        return

    start = line_start if line_start is not None else ln.start
    if start is None:
        start = 0.0

    if line_end is not None and line_end > start + 0.05:
        end = line_end
    elif ln.end is not None and ln.end > start + 0.05:
        end = ln.end
    else:
        end = start + max(1.0, 0.2 * n_words)

    min_word_duration = 0.12
    inter_word_gap = 0.04
    min_line_span = n_words * min_word_duration + max(0, n_words - 1) * inter_word_gap
    span = max(end - start, min_line_span)
    end = start + span

    raw_weights = [max(sum(ch.isalnum() for ch in w), 1) for w in ln.words]
    shaped_weights = [math.sqrt(float(w)) for w in raw_weights]
    weight_sum = sum(shaped_weights)
    if weight_sum <= 0:
        shaped_weights = [1.0] * n_words
        weight_sum = float(n_words)

    available = span - max(0, n_words - 1) * inter_word_gap
    base_floor = min_word_duration * n_words
    extra = max(0.0, available - base_floor)
    durations = [min_word_duration + extra * (w / weight_sum) for w in shaped_weights]

    starts: List[Optional[float]] = []
    ends: List[Optional[float]] = []
    cursor = start
    for i, dur in enumerate(durations):
        starts.append(cursor)
        word_end = cursor + dur
        if i == n_words - 1:
            word_end = end
        ends.append(word_end)
        cursor = word_end + inter_word_gap

    conf = max(0.2, min(0.5, float(line_confidence) * 0.6 if line_confidence else 0.3))
    ln.word_starts = starts
    ln.word_ends = ends
    ln.word_confidences = [conf] * n_words
