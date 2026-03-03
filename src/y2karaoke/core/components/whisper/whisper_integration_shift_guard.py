"""Shift-guard helpers for baseline constraint decisions."""

from typing import List, Tuple

from ... import models


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def should_apply_baseline_constraint(
    mapped_lines: List[models.Line],
    baseline_lines: List[models.Line],
    *,
    matched_ratio: float,
    line_coverage: float,
    min_global_shift_sec: float = 2.5,
    max_global_shift_sec: float = 12.0,
) -> Tuple[bool, float]:
    """Return (apply_constraint, median_global_shift_sec)."""
    shifts: List[float] = []
    limit = min(len(mapped_lines), len(baseline_lines))
    for idx in range(limit):
        mapped = mapped_lines[idx]
        baseline = baseline_lines[idx]
        if not mapped.words or not baseline.words:
            continue
        shifts.append(mapped.start_time - baseline.start_time)

    median_shift = _median(shifts)
    if (
        matched_ratio >= 0.55
        and line_coverage >= 0.8
        and abs(median_shift) >= min_global_shift_sec
        and abs(median_shift) <= max_global_shift_sec
    ):
        return False, median_shift
    return True, median_shift
