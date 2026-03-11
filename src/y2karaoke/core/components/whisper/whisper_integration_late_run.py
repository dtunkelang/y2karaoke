from __future__ import annotations

from ... import models


def late_run_shift_for_baseline_restore(
    mapped: models.Line,
    baseline: models.Line,
    *,
    min_shift_sec: float,
    max_shift_sec: float,
) -> float | None:
    if not mapped.words or not baseline.words:
        return None
    if len(mapped.words) < 4 or len(baseline.words) < 4:
        return None
    shift = mapped.start_time - baseline.start_time
    if min_shift_sec <= shift <= max_shift_sec:
        return shift
    return None


def late_run_is_restorable(
    run_values: list[float],
    *,
    min_run_length: int,
    max_shift_spread_sec: float,
    min_median_shift_sec: float,
) -> bool:
    if len(run_values) < min_run_length:
        return False
    if max(run_values) - min(run_values) > max_shift_spread_sec:
        return False
    sorted_values = sorted(run_values)
    median_shift = sorted_values[len(sorted_values) // 2]
    return median_shift >= min_median_shift_sec
