"""Surrogate timing helpers for unresolved shared-visibility lyric lines."""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple, cast

import numpy as np

from ..models import TargetLine


def _line_has_assigned_word_timing(ln: TargetLine) -> bool:
    return bool(ln.word_starts and any(s is not None for s in ln.word_starts))


def _cluster_unresolved_visibility_lines(
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


def _find_prev_end_floor(
    line_order: List[TargetLine], first_idx: int
) -> Optional[float]:
    for idx in range(first_idx - 1, -1, -1):
        prev = line_order[idx]
        if prev.word_ends and prev.word_ends[-1] is not None:
            return float(prev.word_ends[-1])
        if prev.end is not None:
            return float(prev.end)
    return None


def _find_next_start_cap(
    line_order: List[TargetLine],
    last_idx: int,
    prev_end_floor: Optional[float],
) -> Optional[float]:
    for idx in range(last_idx + 1, len(line_order)):
        nxt = line_order[idx]
        cand: Optional[float] = None
        if nxt.word_starts and nxt.word_starts[0] is not None:
            cand = float(nxt.word_starts[0])
        elif nxt.visibility_start is not None:
            cand = float(nxt.visibility_start)
        elif nxt.start is not None:
            cand = float(nxt.start)
        if cand is None:
            continue
        if prev_end_floor is None or cand > prev_end_floor + 0.2:
            return cand
    return None


def _assign_surrogate_cluster_timings(
    cluster: List[TargetLine],
    *,
    prev_end_floor: Optional[float],
    next_start_cap: Optional[float],
    onset_hints: Optional[Dict[int, float]] = None,
    assign_line_level_word_timings_fn: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    block_start = max(
        min(float(ln.start) for ln in cluster),
        min(
            float(ln.visibility_start)
            for ln in cluster
            if ln.visibility_start is not None
        ),
    )
    if prev_end_floor is not None:
        block_start = max(block_start, prev_end_floor)

    block_end = (
        max(float(ln.visibility_end) for ln in cluster if ln.visibility_end is not None)
        + 1.0
    )
    if next_start_cap is not None:
        block_end = min(block_end, next_start_cap)

    n = len(cluster)
    min_line = 0.7
    line_gap = 0.2
    min_span = n * min_line + max(0, n - 1) * line_gap
    if block_end < block_start + min_span:
        block_end = block_start + min_span

    lead_slack = 0.0
    if not onset_hints and n == 2:
        long_vis = all(
            (ln.visibility_start is not None and ln.visibility_end is not None)
            and (float(ln.visibility_end) - float(ln.visibility_start)) >= 6.0
            for ln in cluster
        )
        if long_vis:
            max_slack = max(0.0, (block_end - block_start) - min_span)
            lead_slack = min(1.5, 0.6 * max_slack)

    span = block_end - (block_start + lead_slack)
    available = max(min_span, span) - max(0, n - 1) * line_gap
    weights = [math.sqrt(float(max(len(ln.words), 1))) for ln in cluster]
    w_sum = sum(weights) if sum(weights) > 0 else float(n)
    extra = max(0.0, available - n * min_line)
    durations = [min_line + extra * (w / w_sum) for w in weights]

    starts: List[float] = []
    cursor = block_start + lead_slack
    for dur in durations:
        starts.append(cursor)
        cursor += dur + line_gap

    if onset_hints:
        adjusted: List[float] = []
        prev_end = block_start - line_gap
        for i, (ln, dur) in enumerate(zip(cluster, durations)):
            s = starts[i]
            hint = onset_hints.get(id(ln))
            if hint is not None:
                s = max(s, float(hint))
            s = max(s, prev_end + line_gap)

            remaining = len(cluster) - i - 1
            min_tail = remaining * min_line + max(0, remaining) * line_gap
            max_s = block_end - min_tail - dur
            s = min(s, max_s)
            adjusted.append(s)
            prev_end = s + dur
        starts = adjusted

    for ln, line_start, dur in zip(cluster, starts, durations):
        line_end = line_start + dur
        assign_line_level_word_timings_fn(ln, line_start, line_end, 0.4)


def _assign_surrogate_timings_for_unresolved_overlap_blocks(
    g_jobs: List[Tuple[TargetLine, float, float]],
    *,
    line_has_assigned_word_timing_fn: Callable[[TargetLine], bool],
    assign_surrogate_cluster_timings_fn: Callable[..., None],
    collect_onset_hints_fn: Optional[
        Callable[[List[TargetLine]], Dict[int, float]]
    ] = None,
) -> None:
    unresolved = [
        ln
        for ln, _, _ in g_jobs
        if ln.visibility_start is not None
        and ln.visibility_end is not None
        and (float(ln.visibility_end) - float(ln.visibility_start)) >= 2.5
        and not line_has_assigned_word_timing_fn(ln)
    ]
    if len(unresolved) < 2:
        return

    clusters = _cluster_unresolved_visibility_lines(unresolved)
    line_order = [ln for ln, _, _ in g_jobs]
    line_pos = {id(ln): idx for idx, ln in enumerate(line_order)}
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        cluster.sort(key=lambda ln: (line_pos.get(id(ln), 10**9), float(ln.y)))

        first_idx = min(line_pos.get(id(ln), 10**9) for ln in cluster)
        last_idx = max(line_pos.get(id(ln), -1) for ln in cluster)
        prev_end_floor = _find_prev_end_floor(line_order, first_idx)
        next_start_cap = _find_next_start_cap(line_order, last_idx, prev_end_floor)
        onset_hints = (
            collect_onset_hints_fn(cluster) if collect_onset_hints_fn else None
        )
        assign_surrogate_cluster_timings_fn(
            cluster,
            prev_end_floor=prev_end_floor,
            next_start_cap=next_start_cap,
            onset_hints=onset_hints,
        )
