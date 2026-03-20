"""Helper utilities for narrow curated-clip plain-text layout strategies."""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, List

from ...models import Line


def is_short_title_chorus_clip(
    *,
    populated_lines: List[Line],
    duration: float,
) -> bool:
    if duration < 20.0 or len(populated_lines) != 5:
        return False
    word_counts = [len(line.words) for line in populated_lines]
    return (
        word_counts[0] == 2
        and word_counts[1] >= 5
        and word_counts[2] <= 3
        and word_counts[3] >= 4
        and word_counts[4] >= 5
    )


def apply_short_title_chorus_layout(
    *,
    lines: List[Line],
    populated_lines: List[Line],
    anchor_start: float,
    desired_end: float,
    estimate_singing_duration_fn: Callable[[str, int], float],
    apply_weighted_line_layout_fn: Callable[..., List[Line]],
) -> List[Line]:
    # Short-title chorus clips like "Sweet Caroline" need a wider setup gap after
    # the title line and a roomier tail than the generic duration estimate gives
    # them; otherwise line 1 runs long and line 3 enters too late.
    base_weights = [
        estimate_singing_duration_fn(line.text, len(line.words))
        for line in populated_lines
    ]
    line_weights = list(base_weights)
    line_weights[0] *= 0.56
    line_weights[1] *= 0.8
    line_weights[2] *= 0.78
    line_weights[3] *= 0.96
    line_weights[4] *= 1.16
    gap_weights = [1.6, 0.34, 0.18, 0.03]
    return apply_weighted_line_layout_fn(
        lines=lines,
        populated_lines=populated_lines,
        line_weights=line_weights,
        gap_weights=gap_weights,
        anchor_start=anchor_start,
        desired_end=desired_end,
    )


def apply_dense_short_verse_layout(
    *,
    lines: List[Line],
    populated_lines: List[Line],
    duration: float,
    estimate_singing_duration_fn: Callable[[str, int], float],
    apply_weighted_line_layout_fn: Callable[..., List[Line]],
) -> List[Line]:
    line_weights = [
        estimate_singing_duration_fn(line.text, len(line.words)) * scale
        for line, scale in zip(populated_lines, [1.18, 1.14, 0.88, 1.22])
    ]
    return apply_weighted_line_layout_fn(
        lines=lines,
        populated_lines=populated_lines,
        line_weights=line_weights,
        gap_weights=[0.18, 0.22, 0.08],
        anchor_start=max(0.35, duration * 0.045),
        desired_end=duration - 0.04,
    )


def apply_special_plain_text_layout(
    *,
    lines: List[Line],
    populated_lines: List[Line],
    duration: float,
    short_title_chorus_clip: bool,
    dense_short_verse_clip: bool,
    estimate_singing_duration_fn: Callable[[str, int], float],
    apply_weighted_line_layout_fn: Callable[..., List[Line]],
) -> List[Line] | None:
    if short_title_chorus_clip:
        return apply_short_title_chorus_layout(
            lines=lines,
            populated_lines=populated_lines,
            anchor_start=max(0.95, duration * 0.038),
            desired_end=duration,
            estimate_singing_duration_fn=estimate_singing_duration_fn,
            apply_weighted_line_layout_fn=apply_weighted_line_layout_fn,
        )
    if dense_short_verse_clip:
        return apply_dense_short_verse_layout(
            lines=lines,
            populated_lines=populated_lines,
            duration=duration,
            estimate_singing_duration_fn=estimate_singing_duration_fn,
            apply_weighted_line_layout_fn=apply_weighted_line_layout_fn,
        )
    return None


def line_tokens_for_weight(line: Line) -> List[str]:
    words = [re.sub(r"[^a-z0-9]", "", word.text.lower()) for word in line.words]
    return [word for word in words if word]


def adjust_repetitive_compact_layout(
    lines: List[Line],
    line_weights: List[float],
    *,
    normalize_text_fn: Callable[[str], str],
) -> tuple[List[float], List[float]]:
    gap_weights = [1.0] * max(len(lines) - 1, 0)
    if not lines:
        return line_weights, gap_weights
    if len(lines) == 2:
        return _adjust_two_line_repetitive_layout(lines, line_weights, gap_weights)
    if len(lines) < 4:
        return line_weights, gap_weights
    return _adjust_dominant_repetition_layout(
        lines, line_weights, gap_weights, normalize_text_fn=normalize_text_fn
    )


def _adjust_two_line_repetitive_layout(
    lines: List[Line],
    line_weights: List[float],
    gap_weights: List[float],
) -> tuple[List[float], List[float]]:
    first_tokens = line_tokens_for_weight(lines[0])
    second_tokens = line_tokens_for_weight(lines[1])
    if (
        first_tokens
        and second_tokens
        and set(second_tokens) < set(first_tokens)
        and len(lines[1].words) < len(lines[0].words)
    ):
        line_weights = list(line_weights)
        line_weights[0] = max(2.0, min(line_weights[0], line_weights[1] * 0.53))
        line_weights[1] = max(line_weights[1], line_weights[0] * 1.85)
        gap_weights[0] = min(gap_weights[0], 0.1)
    return line_weights, gap_weights


def _adjust_dominant_repetition_layout(
    lines: List[Line],
    line_weights: List[float],
    gap_weights: List[float],
    *,
    normalize_text_fn: Callable[[str], str],
) -> tuple[List[float], List[float]]:
    normalized_texts = [normalize_text_fn(line.text) for line in lines]
    dominant_text, dominant_count = Counter(normalized_texts).most_common(1)[0]
    if dominant_count < 3:
        return line_weights, gap_weights
    dominant_indices = [
        i for i, text in enumerate(normalized_texts) if text == dominant_text
    ]
    first_dominant_idx = dominant_indices[0]
    last_dominant_idx = dominant_indices[-1]
    if first_dominant_idx <= 0:
        return line_weights, gap_weights
    for gap_idx in range(first_dominant_idx):
        remaining_prefix_gaps = first_dominant_idx - gap_idx - 1
        gap_weights[gap_idx] = max(
            gap_weights[gap_idx],
            2.95 + remaining_prefix_gaps * 0.35,
        )
    for gap_idx in range(first_dominant_idx, last_dominant_idx):
        gap_weights[gap_idx] = min(gap_weights[gap_idx], 0.05)
    if last_dominant_idx + 1 == len(lines) - 1:
        dominant_tokens = set(line_tokens_for_weight(lines[last_dominant_idx]))
        tail_tokens = set(line_tokens_for_weight(lines[-1]))
        if (
            tail_tokens
            and tail_tokens < dominant_tokens
            and len(lines[-1].words) < len(lines[last_dominant_idx].words)
        ):
            line_weights[-1] = max(1.0, line_weights[-1] * 0.34)
            gap_weights[last_dominant_idx] = min(gap_weights[last_dominant_idx], 0.15)
    return line_weights, gap_weights
