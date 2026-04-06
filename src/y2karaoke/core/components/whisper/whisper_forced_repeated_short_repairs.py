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


def _repeated_suffix_length(tokens: list[str], *, min_phrase_tokens: int) -> int:
    max_phrase_tokens = len(tokens) // 2
    for phrase_len in range(max_phrase_tokens, min_phrase_tokens - 1, -1):
        if tokens[-2 * phrase_len : -phrase_len] == tokens[-phrase_len:]:
            return phrase_len
    return 0


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


def _eligible_leading_repeated_suffix_shape(
    *,
    base0: models.Line,
    base1: models.Line,
    forced0: models.Line,
    min_phrase_tokens: int,
    max_total_words: int,
    min_tail_gain_sec: float,
    max_tail_gain_sec: float,
    min_following_gap_sec: float,
    max_start_drift_sec: float,
    max_tail_duration_ratio: float,
) -> tuple[bool, int]:
    if not base0.words or not base1.words or not forced0.words:
        return False, 0
    if len(base0.words) != len(forced0.words) or len(base0.words) > max_total_words:
        return False, 0

    base_tokens = _normalize_tokens(base0.text)
    forced_tokens = _normalize_tokens(forced0.text)
    if not base_tokens or base_tokens != forced_tokens:
        return False, 0

    repeated_suffix_len = _repeated_suffix_length(
        base_tokens,
        min_phrase_tokens=min_phrase_tokens,
    )
    if repeated_suffix_len <= 0:
        return False, 0

    start_drift = abs(forced0.start_time - base0.start_time)
    tail_gain = base0.end_time - forced0.end_time
    if start_drift > max_start_drift_sec:
        return False, 0
    if tail_gain < min_tail_gain_sec or tail_gain > max_tail_gain_sec:
        return False, 0
    if forced0_duration := (forced0.end_time - forced0.start_time):
        base0_duration = base0.end_time - base0.start_time
        if (
            base0_duration > 0
            and forced0_duration >= base0_duration * max_tail_duration_ratio
        ):
            return False, 0
    if base1.start_time - base0.end_time < min_following_gap_sec:
        return False, 0
    return True, repeated_suffix_len


def restore_leading_repeated_suffix_tails_from_baseline(
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    *,
    min_phrase_tokens: int = 3,
    max_total_words: int = 10,
    min_tail_gain_sec: float = 1.2,
    max_tail_gain_sec: float = 3.5,
    min_following_gap_sec: float = 0.2,
    max_start_drift_sec: float = 0.4,
    max_tail_duration_ratio: float = 0.75,
    min_gap: float = 0.05,
) -> tuple[List[models.Line], int]:
    if len(baseline_lines) < 2 or len(forced_lines) < 2:
        return forced_lines, 0

    base0, base1 = baseline_lines[:2]
    forced0 = forced_lines[0]
    eligible, repeated_suffix_len = _eligible_leading_repeated_suffix_shape(
        base0=base0,
        base1=base1,
        forced0=forced0,
        min_phrase_tokens=min_phrase_tokens,
        max_total_words=max_total_words,
        min_tail_gain_sec=min_tail_gain_sec,
        max_tail_gain_sec=max_tail_gain_sec,
        min_following_gap_sec=min_following_gap_sec,
        max_start_drift_sec=max_start_drift_sec,
        max_tail_duration_ratio=max_tail_duration_ratio,
    )
    if not eligible:
        return forced_lines, 0

    tail_start_idx = len(base0.words) - repeated_suffix_len
    if tail_start_idx <= 0:
        return forced_lines, 0
    target_end = min(base0.end_time, base1.start_time - min_gap)
    base_tail_start = base0.words[tail_start_idx].start_time
    if target_end <= base_tail_start + 0.05:
        return forced_lines, 0
    prev_forced_end = forced0.words[tail_start_idx - 1].end_time
    if base_tail_start <= prev_forced_end + min_gap:
        return forced_lines, 0

    repaired_words = list(forced0.words[:tail_start_idx])
    baseline_tail_duration = base0.end_time - base_tail_start
    available_tail_duration = target_end - base_tail_start
    if baseline_tail_duration <= 0 or available_tail_duration <= 0:
        return forced_lines, 0
    scale = available_tail_duration / baseline_tail_duration
    for word in base0.words[tail_start_idx:]:
        repaired_words.append(
            models.Word(
                text=word.text,
                start_time=base_tail_start
                + (word.start_time - base_tail_start) * scale,
                end_time=base_tail_start + (word.end_time - base_tail_start) * scale,
                singer=word.singer,
            )
        )

    repaired = list(forced_lines)
    repaired[0] = models.Line(words=repaired_words, singer=forced0.singer)
    return repaired, 1
