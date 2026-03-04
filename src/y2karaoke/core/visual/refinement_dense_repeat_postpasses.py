"""Repeated-line and sparse-run retiming helpers for dense visibility runs."""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

from ..models import TargetLine


def promote_unresolved_first_repeated_lines(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
    *,
    line_start_fn: Callable[[TargetLine], Optional[float]],
    line_end_fn: Callable[[TargetLine], Optional[float]],
    canonical_line_text_fn: Callable[[TargetLine], str],
    assign_line_level_word_timings_fn: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    """Backfill early repeated-line starts when only later repeats were resolved."""
    line_order = [ln for ln, _, _ in g_jobs]
    n = len(line_order)
    if n < 2:
        return

    for i, ln in enumerate(line_order):
        if line_start_fn(ln) is not None:
            continue
        if ln.visibility_start is None or ln.visibility_end is None:
            continue
        if len(ln.words) < 3:
            continue
        text = canonical_line_text_fn(ln)
        if not text:
            continue

        later_idx: Optional[int] = None
        later_start: Optional[float] = None
        later_end: Optional[float] = None
        for j in range(i + 1, n):
            cand = line_order[j]
            if canonical_line_text_fn(cand) != text:
                continue
            s = line_start_fn(cand)
            if s is None:
                continue
            later_idx = j
            later_start = float(s)
            e = line_end_fn(cand)
            later_end = float(e) if e is not None else None
            break
        if later_idx is None or later_start is None:
            continue

        vis_start = float(ln.visibility_start)
        vis_end = float(ln.visibility_end)
        if (later_start - vis_start) < 3.0:
            continue

        later_line = line_order[later_idx]
        later_vis_start = (
            float(later_line.visibility_start)
            if later_line.visibility_start is not None
            else later_start
        )
        later_vis_end = (
            float(later_line.visibility_end)
            if later_line.visibility_end is not None
            else (later_end if later_end is not None else later_start + 1.0)
        )
        overlap = min(vis_end, later_vis_end) - max(vis_start, later_vis_start)
        if overlap < 0.6:
            continue

        prev_end: Optional[float] = None
        for k in range(i - 1, -1, -1):
            pe = line_end_fn(line_order[k])
            if pe is not None:
                prev_end = float(pe)
                break
        start = vis_start + 0.05
        if prev_end is not None:
            start = max(start, prev_end + 0.15)
        if start >= later_start - 0.4:
            continue

        ref_dur = (
            later_end - later_start
            if (later_end is not None and later_end > later_start)
            else max(0.8, 0.22 * len(ln.words))
        )
        dur = min(max(ref_dur, 0.7), 2.0)
        end = start + dur

        next_start: Optional[float] = None
        for k in range(i + 1, n):
            ns = line_start_fn(line_order[k])
            if ns is not None:
                next_start = float(ns)
                break
        if next_start is not None:
            end = min(end, next_start - 0.15)
        if end <= start + 0.55:
            end = min(later_start - 0.15, start + 0.85)
        if end <= start + 0.45:
            continue

        assign_line_level_word_timings_fn(ln, start, end, 0.4)


def compress_overlong_sparse_line_timings(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
    *,
    line_start_fn: Callable[[TargetLine], Optional[float]],
    assign_line_level_word_timings_fn: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    """Compress lines with sparse, overlong word timings inside shared-visibility blocks."""
    line_order = [ln for ln, _, _ in g_jobs]
    n = len(line_order)
    if n < 2:
        return

    for i, ln in enumerate(line_order):
        if not ln.word_starts:
            continue
        known = [float(s) for s in ln.word_starts if s is not None]
        if len(known) < 2:
            continue
        start = known[0]
        span = known[-1] - start
        max_reasonable = max(2.6, 0.6 * float(max(len(ln.words), 1)))
        if span <= max_reasonable:
            continue

        if ln.visibility_start is None or ln.visibility_end is None:
            continue
        vis_start = float(ln.visibility_start)
        vis_end = float(ln.visibility_end)
        has_overlap_neighbor = False
        for j in range(max(0, i - 2), min(n, i + 3)):
            if j == i:
                continue
            other = line_order[j]
            if other.visibility_start is None or other.visibility_end is None:
                continue
            ov = min(vis_end, float(other.visibility_end)) - max(
                vis_start, float(other.visibility_start)
            )
            if ov >= 0.8:
                has_overlap_neighbor = True
                break
        if not has_overlap_neighbor:
            continue

        target_dur = min(max(1.0, 0.32 * float(max(len(ln.words), 1)) + 0.8), 2.8)
        end = start + target_dur
        end = min(end, vis_end + 0.4)

        next_start: Optional[float] = None
        for j in range(i + 1, n):
            ns = line_start_fn(line_order[j])
            if ns is not None:
                next_start = float(ns)
                break
        if next_start is not None:
            end = min(end, next_start - 0.15)
        if end <= start + 0.55:
            continue

        assign_line_level_word_timings_fn(ln, start, end, 0.38)


def rebalance_early_lead_shared_visibility_runs(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
    *,
    line_start_fn: Callable[[TargetLine], Optional[float]],
    line_end_fn: Callable[[TargetLine], Optional[float]],
    assign_line_level_word_timings_fn: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    line_order = [ln for ln, _, _ in g_jobs]
    n = len(line_order)
    i = 0
    while i < n:
        base = line_order[i]
        if base.visibility_start is None or base.visibility_end is None:
            i += 1
            continue
        run = [base]
        j = i + 1
        while j < n:
            ln = line_order[j]
            if ln.visibility_start is None or ln.visibility_end is None:
                break
            if (
                abs(float(ln.visibility_start) - float(base.visibility_start)) <= 1.0
                and abs(float(ln.visibility_end) - float(base.visibility_end)) <= 2.0
            ):
                run.append(ln)
                j += 1
                continue
            break

        if len(run) >= 3:
            starts = [line_start_fn(ln) for ln in run]
            ends = [line_end_fn(ln) for ln in run]
            if all(s is not None for s in starts) and all(e is not None for e in ends):
                starts_f = [float(s) for s in starts if s is not None]
                vis_start = float(base.visibility_start)
                if starts_f[0] < vis_start - 0.8:
                    lengths = [max(len(ln.words), 1) for ln in run]
                    if max(lengths) <= 1.5 * min(lengths):
                        prev_end: Optional[float] = None
                        if i > 0:
                            prev_end = line_end_fn(line_order[i - 1])
                        next_start: Optional[float] = None
                        if j < n:
                            next_start = line_start_fn(line_order[j])
                        start0 = max(
                            vis_start - 1.0,
                            (
                                (prev_end + 0.2)
                                if prev_end is not None
                                else vis_start - 1.0
                            ),
                        )
                        end_cap = float(base.visibility_end) + 0.1
                        if next_start is not None:
                            end_cap = min(end_cap, next_start - 0.2)
                        if end_cap > start0 + 2.4:
                            m = len(run)
                            line_gap = 0.2
                            min_line = 0.7
                            min_span = m * min_line + (m - 1) * line_gap
                            if end_cap >= start0 + min_span:
                                avail = (end_cap - start0) - (m - 1) * line_gap
                                weights = [
                                    math.sqrt(float(max(len(ln.words), 1)))
                                    for ln in run
                                ]
                                w_sum = sum(weights) if sum(weights) > 0 else float(m)
                                extra = max(0.0, avail - m * min_line)
                                durs = [min_line + extra * (w / w_sum) for w in weights]
                                cur = start0
                                for ln, dur in zip(run, durs):
                                    assign_line_level_word_timings_fn(
                                        ln,
                                        cur,
                                        cur + dur,
                                        0.42,
                                    )
                                    cur += dur + line_gap

        i = j if j > i else i + 1
