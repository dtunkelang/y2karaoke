"""Repairs for sparse accepted forced-alignment followup lines."""

from __future__ import annotations

from typing import Any, List

from ... import models


def _segment_mean_score(segment: Any) -> float | None:
    if not isinstance(segment, dict):
        return None
    seg_words = segment.get("words")
    if not isinstance(seg_words, list) or not seg_words:
        return None
    scores: list[float] = []
    for seg_word in seg_words:
        if not isinstance(seg_word, dict):
            return None
        score = seg_word.get("score")
        if not isinstance(score, (int, float)):
            return None
        scores.append(float(score))
    if not scores:
        return None
    return sum(scores) / len(scores)


def _restore_candidate_from_source(
    *,
    baseline_line: models.Line,
    prev_line: models.Line,
    next_line: models.Line | None,
    min_gap_sec: float,
) -> models.Line | None:
    start_time = max(baseline_line.start_time, prev_line.end_time + min_gap_sec)
    end_time = baseline_line.end_time
    if next_line is not None and next_line.words:
        end_time = min(end_time, next_line.start_time - min_gap_sec)
    if end_time <= start_time + 0.2:
        return None

    baseline_duration = baseline_line.end_time - baseline_line.start_time
    if baseline_duration <= 0:
        return None
    scale = (end_time - start_time) / baseline_duration
    if scale <= 0:
        return None

    return models.Line(
        words=[
            models.Word(
                text=word.text,
                start_time=start_time
                + (word.start_time - baseline_line.start_time) * scale,
                end_time=start_time
                + (word.end_time - baseline_line.start_time) * scale,
                singer=word.singer,
            )
            for word in baseline_line.words
        ],
        singer=baseline_line.singer,
    )


def restore_sparse_forced_followup_lines_from_source(
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    aligned_segments: Any,
    *,
    min_word_count: int = 4,
    max_word_count: int = 6,
    min_prev_duration_sec: float = 4.5,
    min_prev_score: float = 0.68,
    max_line_score: float = 0.52,
    min_duration_shortfall_sec: float = 1.4,
    max_duration_ratio: float = 0.7,
    max_start_lead_sec: float = 0.45,
    min_gap_sec: float = 0.05,
) -> tuple[List[models.Line], int]:
    if not isinstance(aligned_segments, list):
        return forced_lines, 0

    repaired = list(forced_lines)
    restored = 0
    for idx in range(
        1, min(len(baseline_lines), len(forced_lines), len(aligned_segments))
    ):
        baseline_line = baseline_lines[idx]
        forced_line = repaired[idx]
        prev_line = repaired[idx - 1]
        if not baseline_line.words or not forced_line.words or not prev_line.words:
            continue
        word_count = len(forced_line.words)
        if word_count < min_word_count or word_count > max_word_count:
            continue
        baseline_duration = baseline_line.end_time - baseline_line.start_time
        forced_duration = forced_line.end_time - forced_line.start_time
        if baseline_duration <= 0 or forced_duration <= 0:
            continue
        if forced_duration > baseline_duration * max_duration_ratio:
            continue
        if baseline_duration - forced_duration < min_duration_shortfall_sec:
            continue
        if baseline_line.start_time - forced_line.start_time > max_start_lead_sec:
            continue
        if forced_line.start_time > prev_line.end_time + 0.15:
            continue

        prev_score = _segment_mean_score(aligned_segments[idx - 1])
        line_score = _segment_mean_score(aligned_segments[idx])
        if prev_score is None or line_score is None:
            continue
        if prev_score < min_prev_score or line_score > max_line_score:
            continue

        next_line = repaired[idx + 1] if idx + 1 < len(repaired) else None
        restored_line = _restore_candidate_from_source(
            baseline_line=baseline_line,
            prev_line=prev_line,
            next_line=next_line,
            min_gap_sec=min_gap_sec,
        )
        if restored_line is None:
            continue
        repaired[idx] = restored_line
        restored += 1
    return repaired, restored
