from __future__ import annotations

import math
import re
from typing import List, Optional, Tuple, cast

import numpy as np

from ..models import TargetLine
from .refinement_shared_visibility_postpasses import (
    _rebalance_two_followups_after_short_lead as _rebalance_two_followups_after_short_lead_impl,
    _retime_followups_in_short_lead_shared_visibility_runs as _retime_followups_in_short_lead_shared_visibility_runs_impl,
    _retime_large_gaps_with_early_visibility as _retime_large_gaps_with_early_visibility_impl,
)
from .refinement_transition_postpasses import (
    _clamp_line_ends_to_visibility_windows as _clamp_line_ends_to_visibility_windows_impl,
    _pull_lines_earlier_after_visibility_transitions as _pull_lines_earlier_after_visibility_transitions_impl,
    _rebalance_middle_lines_in_four_line_shared_visibility_runs as _rebalance_middle_lines_impl,
    _retime_short_interstitial_lines_between_anchors as _retime_short_interstitial_lines_between_anchors_impl,
)
from .refinement_repetition_postpasses import (
    _pull_late_first_lines_in_alternating_repeated_blocks as _pull_late_first_lines_in_alternating_repeated_blocks_impl,
    _retime_repeated_blocks_with_long_tail_gap as _retime_repeated_blocks_with_long_tail_gap_impl,
)


def _canonical_line_text(ln: TargetLine) -> str:
    """Light normalization for repetition guards in timing postpasses."""
    text = ln.text if ln.text else " ".join(ln.words)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\\s']+", " ", text)
    return re.sub(r"\\s+", " ", text).strip()


def _assign_line_level_word_timings(
    ln: TargetLine,
    line_start: Optional[float],
    line_end: Optional[float],
    line_confidence: float,
) -> None:
    n_words = len(ln.words)
    if n_words == 0:
        ln.word_starts = []
        ln.word_ends = []
        ln.word_confidences = []
        return

    start = line_start if line_start is not None else ln.start
    if start is None:
        start = 0.0

    if line_end is not None and line_end > start + 0.05:
        end = line_end
    elif ln.end is not None and ln.end > start + 0.05:
        end = ln.end
    else:
        end = start + max(1.0, 0.2 * n_words)

    min_word_duration = 0.12
    inter_word_gap = 0.04
    min_line_span = n_words * min_word_duration + max(0, n_words - 1) * inter_word_gap
    span = max(end - start, min_line_span)
    end = start + span

    raw_weights = [max(sum(ch.isalnum() for ch in w), 1) for w in ln.words]
    shaped_weights = [math.sqrt(float(w)) for w in raw_weights]
    weight_sum = sum(shaped_weights)
    if weight_sum <= 0:
        shaped_weights = [1.0] * n_words
        weight_sum = float(n_words)

    available = span - max(0, n_words - 1) * inter_word_gap
    base_floor = min_word_duration * n_words
    extra = max(0.0, available - base_floor)
    durations = [min_word_duration + extra * (w / weight_sum) for w in shaped_weights]

    starts: List[Optional[float]] = []
    ends: List[Optional[float]] = []
    cursor = start
    for i, dur in enumerate(durations):
        starts.append(cursor)
        word_end = cursor + dur
        if i == n_words - 1:
            word_end = end
        ends.append(word_end)
        cursor = word_end + inter_word_gap

    conf = max(0.2, min(0.5, float(line_confidence) * 0.6 if line_confidence else 0.3))
    ln.word_starts = starts
    ln.word_ends = ends
    ln.word_confidences = [conf] * n_words


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


def _line_start(ln: TargetLine) -> Optional[float]:
    if ln.word_starts and ln.word_starts[0] is not None:
        return float(ln.word_starts[0])
    return None


def _line_end(ln: TargetLine) -> Optional[float]:
    if ln.word_ends and ln.word_ends[-1] is not None:
        return float(ln.word_ends[-1])
    if ln.end is not None:
        return float(ln.end)
    return None


def _compute_line_min_start_time(
    ln: TargetLine,
    *,
    last_assigned_start: Optional[float],
    last_assigned_visibility_end: Optional[float],
) -> Optional[float]:
    min_start: Optional[float] = None
    if ln.visibility_start is not None:
        min_start = float(ln.visibility_start)
        if ln.visibility_end is not None:
            vis_span = max(0.0, float(ln.visibility_end) - float(ln.visibility_start))
            if vis_span >= 8.0:
                min_start -= 2.0
            elif vis_span >= 5.0:
                min_start -= 1.0
        min_start = max(0.0, min_start)

    enforce_global_gate = True
    if (
        ln.visibility_start is not None
        and last_assigned_visibility_end is not None
        and float(ln.visibility_start) < (float(last_assigned_visibility_end) - 0.2)
    ):
        enforce_global_gate = False

    if last_assigned_start is not None and enforce_global_gate:
        gate = float(last_assigned_start + 0.05)
        min_start = gate if min_start is None else max(min_start, gate)
    return min_start


def _retime_late_first_lines_in_shared_visibility_blocks(
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    line_order = [ln for ln, _, _ in g_jobs]
    lines = [
        ln
        for ln in line_order
        if ln.visibility_start is not None and ln.visibility_end is not None
    ]
    if len(lines) < 3:
        return

    clusters = _cluster_unresolved_visibility_lines(lines)
    pos = {id(ln): i for i, ln in enumerate(line_order)}
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        cluster.sort(key=lambda ln: pos.get(id(ln), 10**9))
        first = cluster[0]
        first_start = _line_start(first)
        if first_start is None:
            continue
        block_start = min(
            float(ln.visibility_start)
            for ln in cluster
            if ln.visibility_start is not None
        )
        block_end = max(
            float(ln.visibility_end) for ln in cluster if ln.visibility_end is not None
        )
        span = block_end - block_start
        if span > 16.0:
            continue
        # Only nudge when first detected onset is suspiciously late for this block.
        if first_start <= block_start + 1.5:
            continue

        first_pos = pos.get(id(first), 10**9)
        prev_floor: Optional[float] = None
        for i in range(first_pos - 1, -1, -1):
            prev_e = _line_end(line_order[i])
            if prev_e is not None:
                prev_floor = prev_e
                break
        if prev_floor is None:
            # Don't pull the very first lyric block earlier without a prior timing anchor.
            continue
        if prev_floor < block_start - 4.0:
            # Ignore distant prior anchors from intro artifacts/logos.
            continue

        floor = (
            block_start if prev_floor is None else max(block_start, prev_floor + 0.05)
        )
        late_shift = min(2.0, max(0.8, 0.18 * span + 0.3))
        anchor = max(floor, first_start - late_shift)
        if anchor >= first_start - 0.2:
            continue

        next_start: Optional[float] = None
        for ln in cluster[1:]:
            ns = _line_start(ln)
            if ns is not None:
                next_start = ns
                break

        old_end = _line_end(first)
        old_dur = (
            (old_end - first_start)
            if (old_end is not None)
            else max(0.8, 0.25 * len(first.words))
        )
        new_end = anchor + old_dur
        if next_start is not None:
            new_end = min(new_end, next_start - 0.15)
        new_end = max(new_end, anchor + 0.7)
        _assign_line_level_word_timings(first, anchor, new_end, 0.45)


def _retime_compressed_shared_visibility_blocks(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    line_order = [ln for ln, _, _ in g_jobs]
    lines = [
        ln
        for ln in line_order
        if ln.visibility_start is not None
        and ln.visibility_end is not None
        and _line_start(ln) is not None
    ]
    if len(lines) < 3:
        return

    clusters = _cluster_unresolved_visibility_lines(lines)
    pos = {id(ln): i for i, ln in enumerate(line_order)}
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        cluster.sort(key=lambda ln: pos.get(id(ln), 10**9))
        starts = [_line_start(ln) for ln in cluster]
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
            prev_e = _line_end(line_order[i])
            if prev_e is not None:
                prev_floor = prev_e
                break
        if prev_floor is None:
            # Avoid pulling the first lyric block earlier without a prior anchor.
            continue
        if prev_floor < block_start - 4.0:
            # Ignore distant prior anchors from intro artifacts/logos.
            continue
        lookback = min(1.5, 0.25 * span)
        early_block_start = max(0.0, block_start - lookback)
        target_start = max(early_block_start, prev_floor + 0.05)

        last_pos = pos.get(id(cluster[-1]), -1)
        next_start_cap: Optional[float] = None
        for i in range(last_pos + 1, len(line_order)):
            ns = _line_start(line_order[i])
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
        # In compressed-late clusters, avoid over-allocating the first line.
        if len(weights) >= 2:
            weights[0] *= 0.35
        w_sum = sum(weights) if sum(weights) > 0 else float(n)
        extra = max(0.0, available - n * min_line)
        durations = [min_line + extra * (w / w_sum) for w in weights]

        cursor = target_start
        for ln, dur in zip(cluster, durations):
            _assign_line_level_word_timings(ln, cursor, cursor + dur, 0.42)


def _promote_unresolved_first_repeated_lines(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    """Backfill early repeated-line starts when only later repeats were resolved."""
    line_order = [ln for ln, _, _ in g_jobs]
    n = len(line_order)
    if n < 2:
        return

    for i, ln in enumerate(line_order):
        if _line_start(ln) is not None:
            continue
        if ln.visibility_start is None or ln.visibility_end is None:
            continue
        if len(ln.words) < 3:
            continue
        text = _canonical_line_text(ln)
        if not text:
            continue

        later_idx: Optional[int] = None
        later_start: Optional[float] = None
        later_end: Optional[float] = None
        for j in range(i + 1, n):
            cand = line_order[j]
            if _canonical_line_text(cand) != text:
                continue
            s = _line_start(cand)
            if s is None:
                continue
            later_idx = j
            later_start = float(s)
            e = _line_end(cand)
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
            pe = _line_end(line_order[k])
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
            ns = _line_start(line_order[k])
            if ns is not None:
                next_start = float(ns)
                break
        if next_start is not None:
            end = min(end, next_start - 0.15)
        if end <= start + 0.55:
            end = min(later_start - 0.15, start + 0.85)
        if end <= start + 0.45:
            continue

        _assign_line_level_word_timings(ln, start, end, 0.4)


def _compress_overlong_sparse_line_timings(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
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
            ns = _line_start(line_order[j])
            if ns is not None:
                next_start = float(ns)
                break
        if next_start is not None:
            end = min(end, next_start - 0.15)
        if end <= start + 0.55:
            continue

        _assign_line_level_word_timings(ln, start, end, 0.38)


def _retime_large_gaps_with_early_visibility(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    _retime_large_gaps_with_early_visibility_impl(
        g_jobs,
        line_start_fn=_line_start,
        line_end_fn=_line_end,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _retime_followups_in_short_lead_shared_visibility_runs(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    _retime_followups_in_short_lead_shared_visibility_runs_impl(
        g_jobs,
        line_start_fn=_line_start,
        line_end_fn=_line_end,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _rebalance_two_followups_after_short_lead(
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    _rebalance_two_followups_after_short_lead_impl(
        g_jobs,
        line_start_fn=_line_start,
        line_end_fn=_line_end,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _clamp_line_ends_to_visibility_windows(
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    _clamp_line_ends_to_visibility_windows_impl(
        g_jobs,
        line_start_fn=_line_start,
        line_end_fn=_line_end,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _pull_lines_earlier_after_visibility_transitions(
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    _pull_lines_earlier_after_visibility_transitions_impl(
        g_jobs,
        line_start_fn=_line_start,
        line_end_fn=_line_end,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _retime_short_interstitial_lines_between_anchors(
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    _retime_short_interstitial_lines_between_anchors_impl(
        g_jobs,
        line_start_fn=_line_start,
        line_end_fn=_line_end,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _rebalance_middle_lines_in_four_line_shared_visibility_runs(
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    _rebalance_middle_lines_impl(
        g_jobs,
        line_start_fn=_line_start,
        line_end_fn=_line_end,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _rebalance_early_lead_shared_visibility_runs(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
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
            starts = [_line_start(ln) for ln in run]
            ends = [_line_end(ln) for ln in run]
            if all(s is not None for s in starts) and all(e is not None for e in ends):
                starts_f = [float(s) for s in starts if s is not None]
                vis_start = float(base.visibility_start)
                if starts_f[0] < vis_start - 0.8:
                    lengths = [max(len(ln.words), 1) for ln in run]
                    if max(lengths) <= 1.5 * min(lengths):
                        prev_end: Optional[float] = None
                        if i > 0:
                            prev_end = _line_end(line_order[i - 1])
                        next_start: Optional[float] = None
                        if j < n:
                            next_start = _line_start(line_order[j])
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
                                    _assign_line_level_word_timings(
                                        ln,
                                        cur,
                                        cur + dur,
                                        0.42,
                                    )
                                    cur += dur + line_gap

        i = j if j > i else i + 1


def _shrink_overlong_leads_in_dense_shared_visibility_runs(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
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

        if len(run) >= 4:
            starts = [_line_start(ln) for ln in run]
            ends = [_line_end(ln) for ln in run]
            if all(s is not None for s in starts) and all(e is not None for e in ends):
                texts = [_canonical_line_text(ln) for ln in run]
                nonempty = [t for t in texts if t]
                if len(set(nonempty)) < len(nonempty):
                    i = j if j > i else i + 1
                    continue
                starts_f = [float(s) for s in starts if s is not None]
                ends_f = [float(e) for e in ends if e is not None]
                vis_start = float(base.visibility_start)

                # If the whole run starts late relative to a nearby previous anchor,
                # pull the block earlier while respecting visibility floors.
                if i > 0:
                    prev_end = _line_end(line_order[i - 1])
                    if prev_end is not None:
                        vis_gap = vis_start - float(prev_end)
                        target_start = max(float(prev_end) + 0.2, vis_start - 1.0)
                        early_shift = starts_f[0] - target_start
                        if 0.4 <= early_shift <= 1.2 and -0.2 <= vis_gap <= 1.6:
                            max_shift_vis = early_shift
                            for ln, s in zip(run, starts_f):
                                if ln.visibility_start is None:
                                    continue
                                vis_floor = float(ln.visibility_start) - 1.0
                                max_shift_vis = min(max_shift_vis, s - vis_floor)
                            if max_shift_vis >= 0.4:
                                early_shift = min(early_shift, max_shift_vis)
                                starts_f = [s - early_shift for s in starts_f]
                                ends_f = [e - early_shift for e in ends_f]

                lead_dur = ends_f[0] - starts_f[0]
                follow_durs = [e - s for s, e in zip(starts_f[1:], ends_f[1:])]
                dense_follow = all(
                    (starts_f[k + 1] - starts_f[k]) <= 1.5
                    for k in range(1, min(len(starts_f) - 1, 3))
                )
                if (
                    lead_dur >= 2.0
                    and follow_durs
                    and dense_follow
                    and lead_dur >= (max(follow_durs[:2]) + 1.2)
                ):
                    lead_target = max(
                        0.85,
                        min(1.25, 0.22 * float(max(len(base.words), 1)) + 0.2),
                    )
                    new_lead_end = starts_f[0] + lead_target
                    shift = starts_f[1] - (new_lead_end + 0.2)

                    max_shift = shift
                    for ln, s in zip(run[1:], starts_f[1:]):
                        if ln.visibility_start is None:
                            continue
                        vis_floor = float(ln.visibility_start) - 1.0
                        max_shift = min(max_shift, s - vis_floor)
                    if shift >= 0.4 and max_shift >= 0.4:
                        shift = min(shift, max_shift)
                        _assign_line_level_word_timings(
                            base, starts_f[0], new_lead_end, 0.42
                        )
                        for ln, s, e in zip(run[1:], starts_f[1:], ends_f[1:]):
                            _assign_line_level_word_timings(
                                ln, s - shift, e - shift, 0.42
                            )
                elif i > 0:
                    # Persist the early-shift-only adjustment even if lead shrink is skipped.
                    for ln, s, e in zip(run, starts_f, ends_f):
                        _assign_line_level_word_timings(ln, s, e, 0.42)

        i = j if j > i else i + 1


def _retime_dense_runs_after_overlong_lead(
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    line_order = [ln for ln, _, _ in g_jobs]
    for i in range(1, len(line_order) - 3):
        a = line_order[i]
        b = line_order[i + 1]
        c = line_order[i + 2]
        d = line_order[i + 3]
        prev = line_order[i - 1]

        a_s = _line_start(a)
        a_e = _line_end(a)
        b_s = _line_start(b)
        b_e = _line_end(b)
        c_s = _line_start(c)
        c_e = _line_end(c)
        d_s = _line_start(d)
        d_e = _line_end(d)
        p_e = _line_end(prev)
        if None in (a_s, a_e, b_s, b_e, c_s, c_e, d_s, d_e, p_e):
            continue

        a_sf = float(cast(float, a_s))
        a_ef = float(cast(float, a_e))
        b_sf = float(cast(float, b_s))
        b_ef = float(cast(float, b_e))
        c_sf = float(cast(float, c_s))
        c_ef = float(cast(float, c_e))
        d_sf = float(cast(float, d_s))
        d_ef = float(cast(float, d_e))
        p_ef = float(cast(float, p_e))

        lead_dur = a_ef - a_sf
        if lead_dur < 2.0:
            continue
        if abs(b_sf - a_ef) > 0.35:
            continue
        if (c_sf - b_sf) > 2.0 or (d_sf - c_sf) > 2.0:
            continue
        if (b_ef - b_sf) > 2.2 or (c_ef - c_sf) > 2.2 or (d_ef - d_sf) > 2.2:
            continue
        if (a_sf - p_ef) > 2.0:
            continue

        word_counts = [
            max(len(a.words), 1),
            max(len(b.words), 1),
            max(len(c.words), 1),
            max(len(d.words), 1),
        ]
        if max(word_counts) > 1.8 * min(word_counts):
            continue
        texts = [_canonical_line_text(x) for x in [a, b, c, d]]
        nonempty = [t for t in texts if t]
        if len(set(nonempty)) < len(nonempty):
            continue

        target_start = max(p_ef + 0.05, a_sf - 1.2)
        block_shift = min(1.2, max(0.0, a_sf - target_start))
        new_a_start = a_sf - block_shift
        target_lead_dur = max(0.85, min(1.1, 0.2 * float(word_counts[0]) + 0.1))
        new_a_end = new_a_start + target_lead_dur

        follower_shift = b_sf - (new_a_end + 0.1)
        follower_shift = min(1.8, max(0.0, follower_shift))
        if block_shift < 0.2 and follower_shift < 0.2:
            continue

        _assign_line_level_word_timings(a, new_a_start, new_a_end, 0.42)
        for ln, s_old, e_old in (
            (b, b_sf, b_ef),
            (c, c_sf, c_ef),
            (d, d_sf, d_ef),
        ):
            dur = max(0.6, e_old - s_old)
            proposed_start = s_old - follower_shift
            vis_floor: Optional[float] = None
            if ln.visibility_start is not None and ln.visibility_end is not None:
                vis_span = float(ln.visibility_end) - float(ln.visibility_start)
                # Long-lived lines often stay unhighlighted for a noticeable lead-in.
                # Keep aggressive dense-run pull-forwards from snapping to appearance.
                lead_lag_floor = 0.6 if vis_span >= 8.0 else 0.2
                vis_floor = float(ln.visibility_start) + lead_lag_floor
            if vis_floor is not None:
                proposed_start = max(proposed_start, vis_floor)
            _assign_line_level_word_timings(
                ln, proposed_start, proposed_start + dur, 0.42
            )


def _pull_dense_short_runs_toward_previous_anchor(
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    line_order = [ln for ln, _, _ in g_jobs]
    for i in range(1, len(line_order) - 3):
        prev = line_order[i - 1]
        a = line_order[i]
        b = line_order[i + 1]
        c = line_order[i + 2]
        d = line_order[i + 3]

        p_s, p_e = _line_start(prev), _line_end(prev)
        a_s, a_e = _line_start(a), _line_end(a)
        b_s, b_e = _line_start(b), _line_end(b)
        c_s, c_e = _line_start(c), _line_end(c)
        d_s, d_e = _line_start(d), _line_end(d)
        if None in (p_s, p_e, a_s, a_e, b_s, b_e, c_s, c_e, d_s, d_e):
            continue

        p_sf = float(cast(float, p_s))
        p_ef = float(cast(float, p_e))
        a_sf = float(cast(float, a_s))
        a_ef = float(cast(float, a_e))
        b_sf = float(cast(float, b_s))
        b_ef = float(cast(float, b_e))
        c_sf = float(cast(float, c_s))
        c_ef = float(cast(float, c_e))
        d_sf = float(cast(float, d_s))
        d_ef = float(cast(float, d_e))

        prev_dur = p_ef - p_sf
        first_gap = a_sf - p_ef
        if prev_dur < 2.5 or not (0.7 <= first_gap <= 1.6):
            continue
        if any(
            (e - s) > 2.2
            for s, e in [(a_sf, a_ef), (b_sf, b_ef), (c_sf, c_ef), (d_sf, d_ef)]
        ):
            continue
        if not (
            0.8 <= (b_sf - a_sf) <= 2.2
            and 0.8 <= (c_sf - b_sf) <= 2.2
            and 0.8 <= (d_sf - c_sf) <= 2.2
        ):
            continue

        lengths = [max(len(x.words), 1) for x in [a, b, c, d]]
        if max(lengths) > 1.8 * min(lengths):
            continue
        texts = [_canonical_line_text(x) for x in [a, b, c, d]]
        nonempty = [t for t in texts if t]
        # Repeated-text blocks (e.g. repeated chorus/outro lines) can have
        # long on-screen dwell before highlight; avoid compressing them early.
        if len(set(nonempty)) < len(nonempty):
            continue

        shift = min(1.2, max(0.0, first_gap - 0.05))
        if shift < 0.35:
            continue

        _assign_line_level_word_timings(a, a_sf - shift, a_ef - shift, 0.42)
        _assign_line_level_word_timings(b, b_sf - shift, b_ef - shift, 0.42)
        _assign_line_level_word_timings(c, c_sf - shift, c_ef - shift, 0.42)
        _assign_line_level_word_timings(d, d_sf - shift, d_ef - shift, 0.42)


def _retime_repeated_blocks_with_long_tail_gap(
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    _retime_repeated_blocks_with_long_tail_gap_impl(
        g_jobs,
        line_start_fn=_line_start,
        line_end_fn=_line_end,
        canonical_line_text_fn=_canonical_line_text,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _pull_late_first_lines_in_alternating_repeated_blocks(
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    _pull_late_first_lines_in_alternating_repeated_blocks_impl(
        g_jobs,
        line_start_fn=_line_start,
        line_end_fn=_line_end,
        canonical_line_text_fn=_canonical_line_text,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )
