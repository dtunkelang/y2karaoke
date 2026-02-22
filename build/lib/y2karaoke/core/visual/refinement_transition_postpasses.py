"""Transition-oriented timing post-pass helpers."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, cast

from ..models import TargetLine


def _clamp_line_ends_to_visibility_windows(
    g_jobs: List[Tuple[TargetLine, float, float]],
    *,
    line_start_fn: Callable[[TargetLine], Optional[float]],
    line_end_fn: Callable[[TargetLine], Optional[float]],
    assign_line_level_word_timings_fn: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    line_order = [ln for ln, _, _ in g_jobs]
    for ln in line_order:
        if ln.visibility_start is None or ln.visibility_end is None:
            continue
        vis_span = float(ln.visibility_end) - float(ln.visibility_start)
        if vis_span < 4.0:
            continue
        s = line_start_fn(ln)
        e = line_end_fn(ln)
        if s is None or e is None:
            continue
        cap = float(ln.visibility_end) + 0.1
        if e <= cap + 1e-6:
            continue
        new_end = max(s + 0.7, cap)
        if new_end >= e - 0.05:
            continue
        assign_line_level_word_timings_fn(ln, s, new_end, 0.42)


def _pull_lines_earlier_after_visibility_transitions(
    g_jobs: List[Tuple[TargetLine, float, float]],
    *,
    line_start_fn: Callable[[TargetLine], Optional[float]],
    line_end_fn: Callable[[TargetLine], Optional[float]],
    assign_line_level_word_timings_fn: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    line_order = [ln for ln, _, _ in g_jobs]
    for idx in range(1, len(line_order)):
        prev = line_order[idx - 1]
        curr = line_order[idx]
        prev_end = line_end_fn(prev)
        curr_start = line_start_fn(curr)
        if prev_end is None or curr_start is None:
            continue
        if prev.visibility_end is None or curr.visibility_start is None:
            continue

        vis_gap = float(curr.visibility_start) - float(prev.visibility_end)
        if not (0.4 <= vis_gap <= 1.6):
            continue
        vis_lag = curr_start - float(curr.visibility_start)
        if not (0.4 <= vis_lag <= 1.6):
            continue
        timing_gap = curr_start - prev_end
        if timing_gap < 1.0:
            continue

        old_end = line_end_fn(curr)
        old_dur = (
            (old_end - curr_start)
            if old_end is not None
            else max(0.8, 0.25 * len(curr.words))
        )
        target = max(prev_end + 0.2, float(curr.visibility_start) - 1.0)
        new_start = max(target, curr_start - 1.2)
        if new_start >= curr_start - 0.2:
            continue
        new_end = new_start + old_dur
        assign_line_level_word_timings_fn(curr, new_start, new_end, 0.42)


def _retime_short_interstitial_lines_between_anchors(
    g_jobs: List[Tuple[TargetLine, float, float]],
    *,
    line_start_fn: Callable[[TargetLine], Optional[float]],
    line_end_fn: Callable[[TargetLine], Optional[float]],
    assign_line_level_word_timings_fn: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    """Recenter very short bridge lines when they are biased too early."""
    line_order = [ln for ln, _, _ in g_jobs]
    for idx in range(1, len(line_order) - 1):
        prev = line_order[idx - 1]
        curr = line_order[idx]
        nxt = line_order[idx + 1]
        prev_end = line_end_fn(prev)
        curr_start = line_start_fn(curr)
        curr_end = line_end_fn(curr)
        next_start = line_start_fn(nxt)
        if None in (prev_end, curr_start, curr_end):
            continue

        prev_end_f = float(cast(float, prev_end))
        curr_start_f = float(cast(float, curr_start))
        curr_end_f = float(cast(float, curr_end))
        next_start_f = float(next_start) if next_start is not None else None
        if curr_start_f <= prev_end_f + 0.05:
            continue

        n_words = max(len(curr.words), 0)
        curr_dur = curr_end_f - curr_start_f
        if n_words > 2 or curr_dur > 1.2 or len(prev.words) < 4:
            continue

        lead_gap = curr_start_f - prev_end_f
        if lead_gap >= 0.45:
            continue

        target_start = curr_start_f + min(0.85, max(0.45, 0.8 - lead_gap))
        if next_start_f is not None:
            target_start = min(target_start, next_start_f - curr_dur - 0.15)
        if (target_start - curr_start_f) < 0.25:
            continue

        if curr.visibility_start is not None:
            target_start = max(target_start, float(curr.visibility_start) + 0.1)
        if curr.visibility_end is not None:
            target_start = min(
                target_start, float(curr.visibility_end) - curr_dur - 0.05
            )
        if target_start <= curr_start_f + 0.2:
            continue

        assign_line_level_word_timings_fn(
            curr,
            target_start,
            target_start + curr_dur,
            0.42,
        )


def _rebalance_middle_lines_in_four_line_shared_visibility_runs(
    g_jobs: List[Tuple[TargetLine, float, float]],
    *,
    line_start_fn: Callable[[TargetLine], Optional[float]],
    line_end_fn: Callable[[TargetLine], Optional[float]],
    assign_line_level_word_timings_fn: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    """Spread compressed middle lines between stable first/last anchors."""
    line_order = [ln for ln, _, _ in g_jobs]
    for i in range(len(line_order) - 3):
        a = line_order[i]
        b = line_order[i + 1]
        c = line_order[i + 2]
        d = line_order[i + 3]
        a_s, a_e = line_start_fn(a), line_end_fn(a)
        b_s, b_e = line_start_fn(b), line_end_fn(b)
        c_s, c_e = line_start_fn(c), line_end_fn(c)
        d_s, d_e = line_start_fn(d), line_end_fn(d)
        if None in (a_s, a_e, b_s, b_e, c_s, c_e, d_s, d_e):
            continue
        if any(
            ln.visibility_start is None or ln.visibility_end is None
            for ln in (a, b, c, d)
        ):
            continue

        vis_starts = [float(cast(float, ln.visibility_start)) for ln in (a, b, c, d)]
        vis_ends = [float(cast(float, ln.visibility_end)) for ln in (a, b, c, d)]
        if (max(vis_starts) - min(vis_starts)) > 0.8:
            continue
        if (max(vis_ends) - min(vis_ends)) > 3.5:
            continue

        a_sf = float(cast(float, a_s))
        b_sf = float(cast(float, b_s))
        c_sf = float(cast(float, c_s))
        d_sf = float(cast(float, d_s))
        span = d_sf - a_sf
        if span < 3.5:
            continue
        if (c_sf - b_sf) >= 1.1:
            continue
        if (d_sf - c_sf) <= 2.0:
            continue

        counts = [max(len(x.words), 1) for x in (a, b, c, d)]
        if max(counts) > 1.8 * min(counts):
            continue

        dur_b = max(0.7, float(cast(float, b_e)) - b_sf)
        dur_c = max(0.7, float(cast(float, c_e)) - c_sf)
        target_b = a_sf + span / 3.0
        target_c = a_sf + 2.0 * span / 3.0
        target_b = max(target_b, a_sf + 0.45)
        target_c = max(target_c, target_b + 0.45)
        target_c = min(target_c, d_sf - dur_c - 0.1)
        target_b = min(target_b, target_c - dur_b - 0.1)
        if target_b <= b_sf + 0.2 and target_c <= c_sf + 0.2:
            continue

        assign_line_level_word_timings_fn(b, target_b, target_b + dur_b, 0.42)
        assign_line_level_word_timings_fn(c, target_c, target_c + dur_c, 0.42)
