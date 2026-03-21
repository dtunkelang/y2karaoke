"""Helpers for mixed-density chorus plain-text clip layouts."""

from __future__ import annotations

from collections import Counter
from typing import Callable, List, Tuple

from ...models import Line


def is_mixed_density_chorus_clip(
    *,
    lines: List[Line],
    duration: float,
    min_duration_sec: float,
    min_lines: int,
    normalize_text_fn: Callable[[str], str],
) -> bool:
    if duration < min_duration_sec or len(lines) < min_lines:
        return False
    normalized_texts = [normalize_text_fn(line.text) for line in lines]
    counts = Counter(normalized_texts)
    repeated_exact_lines = sum(1 for count in counts.values() if count >= 2)
    word_counts = [len(line.words) for line in lines]
    if repeated_exact_lines < 2:
        return False
    if min(word_counts) > 4 or max(word_counts) < 8:
        return False
    parenthetical_lines = sum(
        1 for line in lines if "(" in line.text and ")" in line.text
    )
    return parenthetical_lines >= 1


def build_mixed_density_chorus_layout(
    lines: List[Line],
    *,
    normalize_text_fn: Callable[[str], str],
    estimate_singing_duration_fn: Callable[[str, int], float],
) -> Tuple[List[float], List[float]]:
    normalized_texts = [normalize_text_fn(line.text) for line in lines]
    counts = Counter(normalized_texts)
    line_weights = [
        _mixed_density_chorus_line_weight(
            line,
            repeated_count=counts[text],
            estimate_singing_duration_fn=estimate_singing_duration_fn,
        )
        for line, text in zip(lines, normalized_texts)
    ]
    gap_weights = [
        _mixed_density_chorus_gap_weight(current=line, next_line=lines[idx + 1])
        for idx, line in enumerate(lines[:-1])
    ]
    _rebalance_mixed_density_chorus_coda(
        lines=lines,
        normalized_texts=normalized_texts,
        counts=counts,
        line_weights=line_weights,
        gap_weights=gap_weights,
    )
    return line_weights, gap_weights


def _mixed_density_chorus_line_weight(
    line: Line,
    *,
    repeated_count: int,
    estimate_singing_duration_fn: Callable[[str, int], float],
) -> float:
    words = len(line.words)
    weight = estimate_singing_duration_fn(line.text, words)
    if repeated_count >= 2 and words >= 8:
        weight *= 1.28
    elif repeated_count >= 2 and words <= 4:
        weight *= 1.02
    if "(" in line.text and ")" in line.text and words <= 7:
        weight *= 0.72
    return max(weight, 1.0)


def _mixed_density_chorus_gap_weight(current: Line, next_line: Line) -> float:
    current_words = len(current.words)
    next_words = len(next_line.words)
    if current_words <= 4 and next_words >= current_words + 5:
        return 0.8
    if "(" in current.text and ")" in current.text and current_words <= 7:
        return 0.35
    if next_words >= 10 and current_words <= 6:
        return 0.25
    if current_words >= 8 and next_words <= 4:
        return 0.11
    return 0.08


def _rebalance_mixed_density_chorus_coda(
    *,
    lines: List[Line],
    normalized_texts: List[str],
    counts: Counter[str],
    line_weights: List[float],
    gap_weights: List[float],
) -> None:
    if len(lines) < 12 or len(gap_weights) != len(lines) - 1:
        return

    word_counts = [len(line.words) for line in lines]
    tail_counts = word_counts[-4:]
    if not (
        5 <= tail_counts[0] <= 7
        and tail_counts[1] >= 9
        and tail_counts[2] <= 4
        and 5 <= tail_counts[3] <= 7
    ):
        return

    parenthetical_indices = [
        idx for idx, line in enumerate(lines) if "(" in line.text and ")" in line.text
    ]
    if len(parenthetical_indices) < 2 or (len(lines) - 5) not in parenthetical_indices:
        return

    repeated_long_indices = [
        idx
        for idx, text in enumerate(normalized_texts)
        if counts[text] >= 2 and 8 <= word_counts[idx] <= 10
    ]
    if len(repeated_long_indices) < 2:
        return

    for idx in repeated_long_indices[:2]:
        line_weights[idx] *= 1.15

    first_short_to_long_index = next(
        (
            idx
            for idx in range(len(lines) - 1)
            if counts[normalized_texts[idx]] >= 2
            and word_counts[idx] <= 4
            and word_counts[idx + 1] >= 10
        ),
        None,
    )
    if first_short_to_long_index is not None:
        gap_weights[first_short_to_long_index] = max(
            gap_weights[first_short_to_long_index], 1.12
        )

    gap_weights[-5] = max(gap_weights[-5], 0.77)
    gap_weights[-4] = max(gap_weights[-4], 0.55)
    gap_weights[-3] = max(gap_weights[-3], 0.2)
