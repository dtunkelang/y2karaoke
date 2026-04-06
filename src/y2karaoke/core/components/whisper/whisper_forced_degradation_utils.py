"""Utility helpers for forced-fallback degradation checks."""

from __future__ import annotations

from typing import List

from ... import models


def _forced_finalize_step_enabled(env_name: str, *, env_get) -> bool:
    raw = env_get(env_name, "").strip().lower()
    return raw not in {"0", "false", "no", "off"} if raw else True


def _count_sustained_line_degradations(
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    *,
    min_baseline_duration_sec: float = 3.5,
    max_duration_ratio: float = 0.6,
) -> tuple[int, int]:
    compared = degraded = 0
    for baseline_line, forced_line in zip(baseline_lines, forced_lines):
        if not baseline_line.words or not forced_line.words:
            continue
        baseline_duration = baseline_line.end_time - baseline_line.start_time
        if baseline_duration < min_baseline_duration_sec:
            continue
        compared += 1
        forced_duration = forced_line.end_time - forced_line.start_time
        if forced_duration < baseline_duration * max_duration_ratio:
            degraded += 1
    return degraded, compared


def _should_rollback_sustained_line_degradation(
    baseline_lines: List[models.Line],
    forced_lines: List[models.Line],
    *,
    min_degraded_lines: int = 2,
    min_degraded_ratio: float = 0.5,
) -> tuple[bool, int, int]:
    degraded, compared = _count_sustained_line_degradations(
        baseline_lines,
        forced_lines,
    )
    if compared == 0:
        return False, degraded, compared
    rollback = (
        degraded >= min_degraded_lines and degraded / compared >= min_degraded_ratio
    )
    return rollback, degraded, compared
