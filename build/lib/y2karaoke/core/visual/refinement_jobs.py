"""Job-window construction/grouping helpers for visual refinement."""

from __future__ import annotations

from typing import List, Tuple

from ..models import TargetLine


def build_line_refinement_jobs(
    target_lines: List[TargetLine],
    *,
    lead_in_sec: float = 1.0,
    tail_sec: float = 1.0,
) -> List[Tuple[TargetLine, float, float]]:
    jobs: List[Tuple[TargetLine, float, float]] = []
    lead = max(0.0, float(lead_in_sec))
    tail = max(0.0, float(tail_sec))
    for ln in target_lines:
        if not ln.word_rois:
            continue
        line_start = ln.start
        line_end = ln.end if ln.end is not None else ln.start + 5.0
        if ln.visibility_start is not None:
            line_start = min(line_start, float(ln.visibility_start))
        if ln.visibility_end is not None:
            line_end = max(line_end, float(ln.visibility_end))
        v_start, v_end = max(0.0, line_start - lead), line_end + tail
        jobs.append((ln, v_start, v_end))
    jobs.sort(key=lambda item: item[1])
    return jobs


def merge_line_refinement_jobs(
    jobs: List[Tuple[TargetLine, float, float]],
    *,
    max_group_duration_sec: float,
) -> List[Tuple[float, float, List[Tuple[TargetLine, float, float]]]]:
    groups: List[Tuple[float, float, List[Tuple[TargetLine, float, float]]]] = []
    for ln, v_start, v_end in jobs:
        if not groups:
            groups.append((v_start, v_end, [(ln, v_start, v_end)]))
            continue

        g_start, g_end, g_jobs = groups[-1]
        merged_end = max(g_end, v_end)
        merged_duration = merged_end - g_start
        if v_start <= g_end and merged_duration <= max_group_duration_sec:
            g_jobs.append((ln, v_start, v_end))
            groups[-1] = (g_start, merged_end, g_jobs)
            continue

        groups.append((v_start, v_end, [(ln, v_start, v_end)]))
    return groups
