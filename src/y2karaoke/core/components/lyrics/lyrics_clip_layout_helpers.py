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
