"""Guard/stat helper utilities for Whisper refinement."""

from __future__ import annotations

from typing import List, Optional, Tuple

from ...models import Line, Word


def clone_lines(lines: List[Line]) -> List[Line]:
    return [
        Line(
            words=[
                Word(
                    text=w.text,
                    start_time=w.start_time,
                    end_time=w.end_time,
                    singer=w.singer,
                )
                for w in line.words
            ],
            singer=line.singer,
        )
        for line in lines
    ]


def long_gap_stats(lines: List[Line], threshold: float = 20.0) -> Tuple[int, float]:
    prev_end: Optional[float] = None
    long_count = 0
    max_gap = 0.0
    for line in lines:
        if not line.words:
            continue
        if prev_end is not None:
            gap = line.start_time - prev_end
            if gap > threshold:
                long_count += 1
            if gap > max_gap:
                max_gap = gap
        prev_end = line.end_time
    return long_count, max_gap


def ordering_inversion_stats(
    lines: List[Line], tolerance: float = 0.01
) -> Tuple[int, float]:
    prev_start: Optional[float] = None
    inversions = 0
    max_drop = 0.0
    for line in lines:
        if not line.words:
            continue
        start = line.start_time
        if prev_start is not None and start < (prev_start - tolerance):
            inversions += 1
            max_drop = max(max_drop, prev_start - start)
        prev_start = start
    return inversions, max_drop
