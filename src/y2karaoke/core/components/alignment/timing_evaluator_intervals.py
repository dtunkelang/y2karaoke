"""Interval and coverage helpers for timing evaluator core."""

from __future__ import annotations

from typing import List, Tuple

from ...models import Line
from .timing_models import AudioFeatures


def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda interval: interval[0])
    merged: List[Tuple[float, float]] = []
    cur_start, cur_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
            continue
        merged.append((cur_start, cur_end))
        cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


def clip_intervals(
    intervals: List[Tuple[float, float]], window_start: float, window_end: float
) -> List[Tuple[float, float]]:
    clipped: List[Tuple[float, float]] = []
    for start, end in intervals:
        s = max(start, window_start)
        e = min(end, window_end)
        if e > s:
            clipped.append((s, e))
    return merge_intervals(clipped)


def complement_intervals(
    window_start: float,
    window_end: float,
    intervals: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    if window_end <= window_start:
        return []
    if not intervals:
        return [(window_start, window_end)]
    clipped = clip_intervals(intervals, window_start, window_end)
    if not clipped:
        return [(window_start, window_end)]
    result: List[Tuple[float, float]] = []
    cursor = window_start
    for start, end in clipped:
        if start > cursor:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < window_end:
        result.append((cursor, window_end))
    return result


def intersect_intervals(
    left: List[Tuple[float, float]],
    right: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    if not left or not right:
        return []
    intersections: List[Tuple[float, float]] = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        a_start, a_end = left[i]
        b_start, b_end = right[j]
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        if end > start:
            intersections.append((start, end))
        if a_end < b_end:
            i += 1
        else:
            j += 1
    return intersections


def total_duration(intervals: List[Tuple[float, float]]) -> float:
    return float(sum(max(0.0, end - start) for start, end in intervals))


def compute_audio_coverage_metrics(
    lines: List[Line],
    audio_features: AudioFeatures,
) -> Tuple[float, float, float, int]:
    lyric_intervals = [
        (line.start_time, line.end_time)
        for line in lines
        if line.words and line.end_time > line.start_time
    ]
    merged_lyrics = merge_intervals(lyric_intervals)

    vocal_end = (
        audio_features.vocal_end
        if audio_features.vocal_end > audio_features.vocal_start
        else audio_features.duration
    )
    if merged_lyrics:
        window_start = max(audio_features.vocal_start, merged_lyrics[0][0] - 1.0)
        window_end = min(vocal_end, merged_lyrics[-1][1] + 1.0)
    else:
        window_start = audio_features.vocal_start
        window_end = vocal_end
    if window_end <= window_start:
        return 0.0, 0.0, 0.0, 0

    clipped_lyrics = clip_intervals(merged_lyrics, window_start, window_end)
    silence = clip_intervals(audio_features.silence_regions, window_start, window_end)
    non_silent = complement_intervals(window_start, window_end, silence)

    lyric_duration = total_duration(clipped_lyrics)
    non_silent_duration = total_duration(non_silent)
    overlap = total_duration(intersect_intervals(clipped_lyrics, non_silent))

    lyric_on_vocal_ratio = overlap / lyric_duration if lyric_duration > 0 else 0.0
    vocal_covered_ratio = (
        overlap / non_silent_duration if non_silent_duration > 0 else 0.0
    )
    lyric_in_silence_ratio = (
        max(0.0, lyric_duration - overlap) / lyric_duration
        if lyric_duration > 0
        else 0.0
    )

    lyric_gaps = complement_intervals(window_start, window_end, clipped_lyrics)
    uncovered = intersect_intervals(non_silent, lyric_gaps)
    uncovered_regions = sum(1 for start, end in uncovered if end - start >= 0.5)
    return (
        float(lyric_on_vocal_ratio),
        float(vocal_covered_ratio),
        float(lyric_in_silence_ratio),
        uncovered_regions,
    )
