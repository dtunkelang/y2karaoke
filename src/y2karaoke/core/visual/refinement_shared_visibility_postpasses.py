"""Shared-visibility timing post-pass helpers."""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple, cast

from ..models import TargetLine


def _retime_large_gaps_with_early_visibility(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
    *,
    line_start_fn: Callable[[TargetLine], Optional[float]],
    line_end_fn: Callable[[TargetLine], Optional[float]],
    assign_line_level_word_timings_fn: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    line_order = [ln for ln, _, _ in g_jobs]
    if len(line_order) < 2:
        return

    for idx in range(len(line_order)):
        curr = line_order[idx]
        curr_start = line_start_fn(curr)
        if curr_start is None:
            continue
        if curr.visibility_start is None or curr.visibility_end is None:
            continue

        vis_start = float(curr.visibility_start)
        vis_end = float(curr.visibility_end)
        vis_span = vis_end - vis_start
        if vis_span < 8.0:
            continue
        if curr_start - vis_start < 3.0:
            continue

        anchor_end: Optional[float] = None
        for j in range(idx - 1, -1, -1):
            prev = line_order[j]
            prev_end = line_end_fn(prev)
            if prev_end is None:
                continue
            prev_vis_end = (
                float(prev.visibility_end)
                if prev.visibility_end is not None
                else prev_end
            )
            delta = vis_start - prev_vis_end
            if -1.0 <= delta <= 2.5:
                anchor_end = prev_end
                break

        if anchor_end is None:
            continue
        gap = curr_start - anchor_end
        if gap > 6.0:
            continue
        if gap < 2.2:
            continue

        old_end = line_end_fn(curr)
        old_dur = (
            (old_end - curr_start)
            if (old_end is not None)
            else max(0.8, 0.25 * len(curr.words))
        )
        new_start = max(anchor_end + 0.2, vis_start + 0.2, curr_start - 2.0)
        if new_start >= curr_start - 0.2:
            continue
        new_end = new_start + old_dur

        next_start: Optional[float] = None
        for j in range(idx + 1, len(line_order)):
            ns = line_start_fn(line_order[j])
            if ns is not None:
                next_start = ns
                break
        if next_start is not None:
            new_end = min(new_end, next_start - 0.15)
        new_end = max(new_end, new_start + 0.7)
        assign_line_level_word_timings_fn(curr, new_start, new_end, 0.45)


def _retime_followups_in_short_lead_shared_visibility_runs(  # noqa: C901
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
            first = run[0]
            second = run[1]
            first_start = line_start_fn(first)
            first_end = line_end_fn(first)
            second_start = line_start_fn(second)
            if (
                first_start is not None
                and first_end is not None
                and second_start is not None
            ):
                first_dur = max(0.0, first_end - first_start)
                if first_dur <= 1.1 and (second_start - first_end) <= 0.35:
                    last = run[-1]
                    last_end = line_end_fn(last)
                    if last_end is not None:
                        next_start: Optional[float] = None
                        for k in range(j, n):
                            ns = line_start_fn(line_order[k])
                            if ns is not None:
                                next_start = ns
                                break
                        cap = (
                            (next_start - 0.2)
                            if next_start is not None
                            else (float(base.visibility_end) + 1.0)
                        )
                        tail_slack = cap - last_end
                        if tail_slack >= 0.7:
                            shift = min(1.0, 0.6 * tail_slack, tail_slack - 0.1)
                            if shift >= 0.25:
                                for ln in run[1:]:
                                    s = line_start_fn(ln)
                                    e = line_end_fn(ln)
                                    if s is None or e is None:
                                        continue
                                    assign_line_level_word_timings_fn(
                                        ln,
                                        s + shift,
                                        e + shift,
                                        0.42,
                                    )

        i = j if j > i else i + 1


def _rebalance_two_followups_after_short_lead(
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
    while i + 2 < n:
        a = line_order[i]
        b = line_order[i + 1]
        c = line_order[i + 2]
        if not _has_visibility_windows(a, b, c):
            i += 1
            continue
        if not _is_shared_visibility_triplet(a, b, c):
            i += 1
            continue

        a_start = line_start_fn(a)
        a_end = line_end_fn(a)
        b_start = line_start_fn(b)
        c_start = line_start_fn(c)
        if a_start is None or a_end is None or b_start is None or c_start is None:
            i += 1
            continue
        if (a_end - a_start) > 1.15:
            i += 1
            continue
        if (c_start - b_start) >= 1.4:
            i += 1
            continue

        next_start = _next_assigned_start(
            line_order, i + 3, line_start_fn=line_start_fn
        )
        cap = (
            (next_start - 0.2)
            if next_start is not None
            else (float(cast(float, a.visibility_end)) + 0.8)
        )
        if cap <= b_start + 2.0:
            i += 1
            continue

        durations = _compute_followup_durations(b, c, b_start=b_start, cap=cap)
        if durations is None:
            i += 1
            continue
        dur_b, dur_c = durations
        gap = 0.2
        s_b = b_start
        s_c = s_b + dur_b + gap
        assign_line_level_word_timings_fn(b, s_b, s_b + dur_b, 0.42)
        assign_line_level_word_timings_fn(c, s_c, s_c + dur_c, 0.42)
        i += 3


def _has_visibility_windows(*lines: TargetLine) -> bool:
    return all(
        line.visibility_start is not None and line.visibility_end is not None
        for line in lines
    )


def _is_shared_visibility_triplet(a: TargetLine, b: TargetLine, c: TargetLine) -> bool:
    a_vs = float(cast(float, a.visibility_start))
    a_ve = float(cast(float, a.visibility_end))
    b_vs = float(cast(float, b.visibility_start))
    b_ve = float(cast(float, b.visibility_end))
    c_vs = float(cast(float, c.visibility_start))
    c_ve = float(cast(float, c.visibility_end))
    return (
        abs(b_vs - a_vs) <= 1.0
        and abs(c_vs - a_vs) <= 1.0
        and abs(b_ve - a_ve) <= 2.0
        and abs(c_ve - a_ve) <= 2.0
    )


def _next_assigned_start(
    lines: List[TargetLine],
    start_idx: int,
    *,
    line_start_fn: Callable[[TargetLine], Optional[float]],
) -> float | None:
    for k in range(start_idx, len(lines)):
        next_start = line_start_fn(lines[k])
        if next_start is not None:
            return next_start
    return None


def _compute_followup_durations(
    b: TargetLine, c: TargetLine, *, b_start: float, cap: float
) -> tuple[float, float] | None:
    gap = 0.2
    avail = (cap - b_start) - gap
    if avail <= 1.4:
        return None
    min_line = 0.7
    extra = max(0.0, avail - 2.0 * min_line)
    w_b = max(1.0, math.sqrt(float(max(len(b.words), 1)))) * 1.8
    w_c = max(1.0, math.sqrt(float(max(len(c.words), 1))))
    w_sum = w_b + w_c
    dur_b = min_line + extra * (w_b / w_sum)
    dur_c = min_line + extra * (w_c / w_sum)
    return dur_b, dur_c
