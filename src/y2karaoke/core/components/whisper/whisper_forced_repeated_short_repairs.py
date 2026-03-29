"""Forced-fallback repairs for repeated short leading lines."""

from __future__ import annotations

import re
from typing import List

from ... import models

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _normalize_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _set_line_end(line: models.Line, end_time: float) -> models.Line:
    if not line.words or end_time <= line.start_time:
        return line
    words = [
        models.Word(
            text=word.text,
            start_time=word.start_time,
            end_time=(end_time if idx == len(line.words) - 1 else word.end_time),
            singer=word.singer,
        )
        for idx, word in enumerate(line.words)
    ]
    return models.Line(words=words, singer=line.singer)


def _eligible_leading_repeated_short_shape(
    *,
    base0: models.Line,
    base1: models.Line,
    base2: models.Line,
    forced0: models.Line,
    forced1: models.Line,
    max_words_per_line: int,
    min_third_line_words: int,
    min_tail_gain_sec: float,
    max_tail_gain_sec: float,
    min_following_gap_sec: float,
    max_start_drift_sec: float,
) -> bool:
    if (
        not base0.words
        or not base1.words
        or not base2.words
        or not forced0.words
        or not forced1.words
    ):
        return False
    if len(base0.words) > max_words_per_line or len(base1.words) > max_words_per_line:
        return False
    if len(base2.words) < min_third_line_words:
        return False

    base0_tokens = _normalize_tokens(base0.text)
    base1_tokens = _normalize_tokens(base1.text)
    forced0_tokens = _normalize_tokens(forced0.text)
    forced1_tokens = _normalize_tokens(forced1.text)
    if not base0_tokens or base0_tokens != base1_tokens:
        return False
    if forced0_tokens != base0_tokens or forced1_tokens != base0_tokens:
        return False

    start_drift = abs(forced0.start_time - base0.start_time)
    tail_gain = base0.end_time - forced0.end_time
    if start_drift > max_start_drift_sec:
        return False
    if tail_gain < min_tail_gain_sec or tail_gain > max_tail_gain_sec:
        return False
    return forced1.start_time - base0.end_time >= min_following_gap_sec


def restore_leading_repeated_short_line_tails_from_baseline(
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    *,
    max_words_per_line: int = 2,
    min_third_line_words: int = 4,
    min_tail_gain_sec: float = 0.18,
    max_tail_gain_sec: float = 0.4,
    min_following_gap_sec: float = 1.4,
    max_start_drift_sec: float = 0.12,
    min_gap: float = 0.05,
) -> tuple[List[models.Line], int]:
    if len(baseline_lines) < 3 or len(forced_lines) < 3:
        return forced_lines, 0

    base0, base1, base2 = baseline_lines[:3]
    forced0, forced1 = forced_lines[:2]
    if not _eligible_leading_repeated_short_shape(
        base0=base0,
        base1=base1,
        base2=base2,
        forced0=forced0,
        forced1=forced1,
        max_words_per_line=max_words_per_line,
        min_third_line_words=min_third_line_words,
        min_tail_gain_sec=min_tail_gain_sec,
        max_tail_gain_sec=max_tail_gain_sec,
        min_following_gap_sec=min_following_gap_sec,
        max_start_drift_sec=max_start_drift_sec,
    ):
        return forced_lines, 0

    target_end = min(base0.end_time, forced1.start_time - min_gap)
    if target_end <= forced0.end_time + 0.01:
        return forced_lines, 0

    repaired = list(forced_lines)
    repaired[0] = _set_line_end(forced0, target_end)
    return repaired, 1
