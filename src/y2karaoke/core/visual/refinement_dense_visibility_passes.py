"""Shared-visibility clustering and early dense-run retiming passes."""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple, cast

import numpy as np

from ..models import TargetLine


def cluster_unresolved_visibility_lines(
    lines: List[TargetLine],
) -> List[List[TargetLine]]:
    clusters: List[List[TargetLine]] = []
    filtered = [
        ln
        for ln in lines
        if ln.visibility_start is not None and ln.visibility_end is not None
    ]
    ordered = sorted(
        filtered,
        key=lambda ln: (cast(float, ln.visibility_start), float(ln.y)),
    )
    for ln in ordered:
        ln_start = cast(float, ln.visibility_start)
        ln_end = cast(float, ln.visibility_end)
        placed = False
        for cluster in clusters:
            starts = [
                float(c.visibility_start)
                for c in cluster
                if c.visibility_start is not None
            ]
            ends = [
                float(c.visibility_end) for c in cluster if c.visibility_end is not None
            ]
            if not starts or not ends:
                continue
            c_start = float(np.median(np.array(starts, dtype=np.float32)))
            c_end = float(np.median(np.array(ends, dtype=np.float32)))
            if abs(ln_start - c_start) <= 2.0 and abs(ln_end - c_end) <= 3.0:
                cluster.append(ln)
                placed = True
                break
        if not placed:
            clusters.append([ln])
    return clusters


def _cluster_visibility_span(cluster: List[TargetLine]) -> tuple[float, float, float]:
    block_start = min(
        float(ln.visibility_start) for ln in cluster if ln.visibility_start is not None
    )
    block_end = max(
        float(ln.visibility_end) for ln in cluster if ln.visibility_end is not None
    )
    return block_start, block_end, block_end - block_start


def _find_previous_end_floor(
    line_order: List[TargetLine],
    *,
    first_pos: int,
    line_end_fn: Callable[[TargetLine], Optional[float]],
) -> Optional[float]:
    for i in range(first_pos - 1, -1, -1):
        prev_e = line_end_fn(line_order[i])
        if prev_e is not None:
            return prev_e
    return None


def _first_non_null_start(
    cluster: List[TargetLine],
    *,
    line_start_fn: Callable[[TargetLine], Optional[float]],
    skip_first: bool = True,
) -> Optional[float]:
    seq = cluster[1:] if skip_first else cluster
    for ln in seq:
        start = line_start_fn(ln)
        if start is not None:
            return start
    return None


def _retime_first_cluster_line(
    first: TargetLine,
    *,
    anchor: float,
    first_start: float,
    next_start: Optional[float],
    line_end_fn: Callable[[TargetLine], Optional[float]],
    assign_line_level_word_timings_fn: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    old_end = line_end_fn(first)
    old_dur = (
        (old_end - first_start)
        if (old_end is not None)
        else max(0.8, 0.25 * len(first.words))
    )
    new_end = anchor + old_dur
    if next_start is not None:
        new_end = min(new_end, next_start - 0.15)
    new_end = max(new_end, anchor + 0.7)
    assign_line_level_word_timings_fn(first, anchor, new_end, 0.45)


def retime_late_first_lines_in_shared_visibility_blocks(
    g_jobs: List[Tuple[TargetLine, float, float]],
    *,
    line_start_fn: Callable[[TargetLine], Optional[float]],
    line_end_fn: Callable[[TargetLine], Optional[float]],
    assign_line_level_word_timings_fn: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    line_order = [ln for ln, _, _ in g_jobs]
    lines = [
        ln
        for ln in line_order
        if ln.visibility_start is not None and ln.visibility_end is not None
    ]
    if len(lines) < 3:
        return

    clusters = cluster_unresolved_visibility_lines(lines)
    pos = {id(ln): i for i, ln in enumerate(line_order)}
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        cluster.sort(key=lambda ln: pos.get(id(ln), 10**9))
        first = cluster[0]
        first_start = line_start_fn(first)
        if first_start is None:
            continue
        block_start, _block_end, span = _cluster_visibility_span(cluster)
        if span > 16.0:
            continue
        if first_start <= block_start + 1.5:
            continue

        first_pos = pos.get(id(first), 10**9)
        prev_floor = _find_previous_end_floor(
            line_order, first_pos=first_pos, line_end_fn=line_end_fn
        )
        if prev_floor is None:
            continue
        if prev_floor < block_start - 4.0:
            continue

        floor = (
            block_start if prev_floor is None else max(block_start, prev_floor + 0.05)
        )
        late_shift = min(2.0, max(0.8, 0.18 * span + 0.3))
        anchor = max(floor, first_start - late_shift)
        if anchor >= first_start - 0.2:
            continue

        next_start = _first_non_null_start(cluster, line_start_fn=line_start_fn)
        _retime_first_cluster_line(
            first,
            anchor=anchor,
            first_start=first_start,
            next_start=next_start,
            line_end_fn=line_end_fn,
            assign_line_level_word_timings_fn=assign_line_level_word_timings_fn,
        )


def retime_compressed_shared_visibility_blocks(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
    *,
    line_start_fn: Callable[[TargetLine], Optional[float]],
    line_end_fn: Callable[[TargetLine], Optional[float]],
    assign_line_level_word_timings_fn: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    line_order = [ln for ln, _, _ in g_jobs]
    lines = [
        ln
        for ln in line_order
        if ln.visibility_start is not None
        and ln.visibility_end is not None
        and line_start_fn(ln) is not None
    ]
    if len(lines) < 3:
        return

    clusters = cluster_unresolved_visibility_lines(lines)
    pos = {id(ln): i for i, ln in enumerate(line_order)}
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        cluster.sort(key=lambda ln: pos.get(id(ln), 10**9))
        starts = [line_start_fn(ln) for ln in cluster]
        if any(s is None for s in starts):
            continue
        starts_f = [float(s) for s in starts if s is not None]
        block_start = min(
            float(ln.visibility_start)
            for ln in cluster
            if ln.visibility_start is not None
        )
        block_end = max(
            float(ln.visibility_end) for ln in cluster if ln.visibility_end is not None
        )
        span = block_end - block_start
        if span <= 0.0 or span > 8.0:
            continue

        late_threshold = block_start + 0.50 * span
        late_count = sum(1 for s in starts_f if s >= late_threshold)
        gap_values = [starts_f[i + 1] - starts_f[i] for i in range(len(starts_f) - 1)]
        max_gap = max(gap_values) if gap_values else 0.0
        has_large_internal_gap = max_gap >= 1.2
        if late_count < max(2, len(cluster) - 2) and not has_large_internal_gap:
            continue

        first_pos = pos.get(id(cluster[0]), 10**9)
        prev_floor: Optional[float] = None
        for i in range(first_pos - 1, -1, -1):
            prev_e = line_end_fn(line_order[i])
            if prev_e is not None:
                prev_floor = prev_e
                break
        if prev_floor is None:
            continue
        if prev_floor < block_start - 4.0:
            continue
        lookback = min(1.5, 0.25 * span)
        early_block_start = max(0.0, block_start - lookback)
        target_start = max(early_block_start, prev_floor + 0.05)

        last_pos = pos.get(id(cluster[-1]), -1)
        next_start_cap: Optional[float] = None
        for i in range(last_pos + 1, len(line_order)):
            ns = line_start_fn(line_order[i])
            if ns is not None:
                next_start_cap = ns
                break
        target_end = block_end + 1.0
        if next_start_cap is not None:
            target_end = min(target_end, next_start_cap - 0.15)

        n = len(cluster)
        min_line = 0.7
        line_gap = 0.2
        min_span = n * min_line + max(0, n - 1) * line_gap
        if target_end < target_start + min_span:
            continue

        available = (target_end - target_start) - max(0, n - 1) * line_gap
        weights = [math.sqrt(float(max(len(ln.words), 1))) for ln in cluster]
        if len(weights) >= 2:
            weights[0] *= 0.35
        w_sum = sum(weights) if sum(weights) > 0 else float(n)
        extra = max(0.0, available - n * min_line)
        durations = [min_line + extra * (w / w_sum) for w in weights]

        cursor = target_start
        for ln, dur in zip(cluster, durations):
            assign_line_level_word_timings_fn(ln, cursor, cursor + dur, 0.42)
