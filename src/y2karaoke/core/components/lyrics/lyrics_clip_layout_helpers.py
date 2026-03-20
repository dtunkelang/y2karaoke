"""Helper utilities for narrow curated-clip plain-text layout strategies."""

from __future__ import annotations

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
    base_weights = [
        estimate_singing_duration_fn(line.text, len(line.words))
        for line in populated_lines
    ]
    line_weights = list(base_weights)
    line_weights[0] *= 0.68
    line_weights[1] *= 0.86
    line_weights[2] *= 0.76
    line_weights[3] *= 1.0
    gap_weights = [1.0, 0.55, 0.22, 0.08]
    return apply_weighted_line_layout_fn(
        lines=lines,
        populated_lines=populated_lines,
        line_weights=line_weights,
        gap_weights=gap_weights,
        anchor_start=anchor_start,
        desired_end=desired_end,
    )
