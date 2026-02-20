"""Persistent-overlap block helpers for visual timing refinement."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from ..models import TargetLine

FrameTriplet = Tuple[float, np.ndarray, np.ndarray]


def apply_persistent_block_highlight_order(
    g_jobs: List[Tuple[TargetLine, float, float]],
    group_frames: List[FrameTriplet],
    group_times: List[float],
    *,
    select_persistent_overlap_lines: Callable[[List[TargetLine]], List[TargetLine]],
    cluster_persistent_lines_by_visibility: Callable[
        [List[TargetLine]], List[List[TargetLine]]
    ],
    assign_cluster_persistent_onsets: Callable[
        [List[TargetLine], List[FrameTriplet], List[float]], None
    ],
) -> None:
    """Override timings for persistent overlapping lines using highlight order."""
    candidates: List[TargetLine] = []
    for ln, _, _ in g_jobs:
        if not ln.word_rois or ln.visibility_start is None or ln.visibility_end is None:
            continue
        if (float(ln.visibility_end) - float(ln.visibility_start)) >= 12.0:
            candidates.append(ln)
    persistent = select_persistent_overlap_lines(candidates)
    if len(persistent) < 3:
        return
    clusters = cluster_persistent_lines_by_visibility(persistent)
    for cluster in clusters:
        assign_cluster_persistent_onsets(cluster, group_frames, group_times)


def assign_cluster_persistent_onsets(
    cluster: List[TargetLine],
    group_frames: List[FrameTriplet],
    group_times: List[float],
    *,
    collect_persistent_block_onset_candidates: Callable[
        [List[TargetLine], List[FrameTriplet], List[float]],
        List[Tuple[TargetLine, Optional[float], float]],
    ],
    assign_line_level_word_timings: Callable[[TargetLine, float, float, float], None],
) -> None:
    if len(cluster) < 3:
        return
    onset_candidates = collect_persistent_block_onset_candidates(
        cluster, group_frames, group_times
    )
    if sum(1 for _, s, c in onset_candidates if s is not None and c >= 0.25) < 2:
        return

    prev_onset: Optional[float] = None
    assigned: List[Tuple[TargetLine, float, float]] = []
    for ln, onset, conf in onset_candidates:
        if onset is None:
            continue
        if prev_onset is not None:
            onset = max(onset, prev_onset + 0.2)
        prev_onset = onset
        assigned.append((ln, onset, conf))
    if len(assigned) < 2:
        return

    for i, (ln, onset, conf) in enumerate(assigned):
        next_onset = assigned[i + 1][1] if i + 1 < len(assigned) else None
        if next_onset is not None and (next_onset - onset) > 6.0:
            next_onset = None
        if next_onset is None:
            surrogate_end = onset + max(2.2, 0.33 * len(ln.words))
            if ln.visibility_end is not None:
                surrogate_end = min(surrogate_end, float(ln.visibility_end) + 0.5)
            next_onset = surrogate_end
        assign_line_level_word_timings(
            ln,
            onset,
            next_onset,
            max(0.35, min(0.75, conf)),
        )


def select_persistent_overlap_lines(candidates: List[TargetLine]) -> List[TargetLine]:
    if len(candidates) < 3:
        return []
    persistent: List[TargetLine] = []
    for ln in candidates:
        if ln.visibility_start is None or ln.visibility_end is None:
            continue
        overlaps = 0
        for other in candidates:
            if other is ln:
                continue
            if other.visibility_start is None or other.visibility_end is None:
                continue
            ov = min(float(ln.visibility_end), float(other.visibility_end)) - max(
                float(ln.visibility_start), float(other.visibility_start)
            )
            if ov >= 8.0:
                overlaps += 1
        if overlaps >= 2:
            persistent.append(ln)
    persistent.sort(key=lambda ln: float(ln.y))
    return persistent


def cluster_persistent_lines_by_visibility(
    lines: List[TargetLine],
) -> List[List[TargetLine]]:
    clusters: List[List[TargetLine]] = []
    ordered = sorted(
        lines, key=lambda ln: (float(ln.visibility_start or 0.0), float(ln.y))
    )
    for ln in ordered:
        if ln.visibility_start is None or ln.visibility_end is None:
            continue
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
            if (
                abs(float(ln.visibility_start) - c_start) <= 2.0
                and abs(float(ln.visibility_end) - c_end) <= 3.0
            ):
                cluster.append(ln)
                placed = True
                break
        if not placed:
            clusters.append([ln])

    for cluster in clusters:
        cluster.sort(key=lambda ln: float(ln.y))
    return clusters


def collect_persistent_block_onset_candidates(
    persistent: List[TargetLine],
    group_frames: List[FrameTriplet],
    group_times: List[float],
    *,
    slice_frames_for_window: Callable[
        [List[FrameTriplet], List[float], float, float], List[FrameTriplet]
    ],
    collect_line_color_values: Callable[
        [TargetLine, List[FrameTriplet], np.ndarray], List[dict[str, object]]
    ],
    estimate_onset_from_visibility_progress: Callable[
        [List[dict[str, object]]], Tuple[Optional[float], float]
    ],
) -> List[Tuple[TargetLine, Optional[float], float]]:
    onset_candidates: List[Tuple[TargetLine, Optional[float], float]] = []
    for ln in persistent:
        if ln.visibility_start is None or ln.visibility_end is None:
            onset_candidates.append((ln, None, 0.0))
            continue
        v_start = max(0.0, float(ln.visibility_start) - 0.5)
        v_end = float(ln.visibility_end) + 0.5
        line_frames = slice_frames_for_window(group_frames, group_times, v_start, v_end)
        if len(line_frames) < 8:
            onset_candidates.append((ln, None, 0.0))
            continue
        c_bg_line = np.mean(
            [
                np.mean(f[1], axis=(0, 1))
                for f in line_frames[: min(10, len(line_frames))]
            ],
            axis=0,
        )
        vals = collect_line_color_values(ln, line_frames, c_bg_line)
        onset, conf = estimate_onset_from_visibility_progress(vals)
        onset_candidates.append((ln, onset, conf))
    return onset_candidates
