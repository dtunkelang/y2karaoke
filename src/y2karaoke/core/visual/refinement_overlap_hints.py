"""Onset-hint helpers for unresolved overlap refinement paths."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..models import TargetLine

FrameTriplet = Tuple[float, np.ndarray, np.ndarray]


def line_has_assigned_word_timing(ln: TargetLine) -> bool:
    return bool(ln.word_starts and any(s is not None for s in ln.word_starts))


def collect_unresolved_line_onset_hints(
    cluster: List[TargetLine],
    group_frames: List[FrameTriplet],
    group_times: List[float],
    *,
    slice_frames_for_window: Callable[
        [List[FrameTriplet], List[float], float, float], List[FrameTriplet]
    ],
    detect_line_highlight_with_confidence: Callable[
        [TargetLine, List[FrameTriplet], np.ndarray, float, bool],
        Tuple[Optional[float], Optional[float], float],
    ],
    collect_line_color_values: Callable[
        [TargetLine, List[FrameTriplet], np.ndarray], List[dict[str, object]]
    ],
    estimate_onset_from_visibility_progress: Callable[
        [List[dict[str, object]]], Tuple[Optional[float], float]
    ],
) -> Dict[int, float]:
    hints: Dict[int, float] = {}
    for ln in cluster:
        if ln.visibility_start is None or ln.visibility_end is None:
            continue
        v_start = max(0.0, float(ln.visibility_start) - 0.5)
        v_end = float(ln.visibility_end) + 0.5
        line_frames = slice_frames_for_window(group_frames, group_times, v_start, v_end)
        if len(line_frames) < 8:
            continue
        c_bg_line = np.mean(
            [
                np.mean(f[1], axis=(0, 1))
                for f in line_frames[: min(10, len(line_frames))]
            ],
            axis=0,
        )
        s, _e, conf = detect_line_highlight_with_confidence(
            ln,
            line_frames,
            c_bg_line,
            float(ln.visibility_start),
            False,
        )
        if s is not None and conf >= 0.45:
            hints[id(ln)] = float(s)
            continue
        vals = collect_line_color_values(ln, line_frames, c_bg_line)
        onset, onset_conf = estimate_onset_from_visibility_progress(vals)
        if onset is not None and onset_conf >= 0.45:
            hints[id(ln)] = float(onset)
    return hints


def estimate_line_onset_hint_in_visibility_window(
    ln: TargetLine,
    group_frames: List[FrameTriplet],
    group_times: List[float],
    *,
    slice_frames_for_window: Callable[
        [List[FrameTriplet], List[float], float, float], List[FrameTriplet]
    ],
    detect_line_highlight_with_confidence: Callable[
        [TargetLine, List[FrameTriplet], np.ndarray, float, bool],
        Tuple[Optional[float], Optional[float], float],
    ],
    collect_line_color_values: Callable[
        [TargetLine, List[FrameTriplet], np.ndarray], List[dict[str, object]]
    ],
    estimate_onset_from_visibility_progress: Callable[
        [List[dict[str, object]]], Tuple[Optional[float], float]
    ],
) -> Optional[float]:
    if ln.visibility_start is None or ln.visibility_end is None:
        return None
    v_start = max(0.0, float(ln.visibility_start) - 0.5)
    v_end = float(ln.visibility_end) + 0.5
    line_frames = slice_frames_for_window(group_frames, group_times, v_start, v_end)
    if len(line_frames) < 8:
        return None
    c_bg_line = np.mean(
        [np.mean(f[1], axis=(0, 1)) for f in line_frames[: min(10, len(line_frames))]],
        axis=0,
    )
    s, _e, conf = detect_line_highlight_with_confidence(
        ln,
        line_frames,
        c_bg_line,
        float(ln.visibility_start),
        False,
    )
    if s is not None and conf >= 0.45:
        return float(s)
    vals = collect_line_color_values(ln, line_frames, c_bg_line)
    onset, onset_conf = estimate_onset_from_visibility_progress(vals)
    if onset is not None and onset_conf >= 0.45:
        return float(onset)
    return None


def maybe_adjust_detected_line_start_with_visibility_hint(
    ln: TargetLine,
    *,
    detected_start: Optional[float],
    detected_end: Optional[float],
    detected_confidence: float,
    group_frames: List[FrameTriplet],
    group_times: List[float],
    estimate_line_onset_hint_in_visibility_window: Callable[
        [TargetLine, List[FrameTriplet], List[float]], Optional[float]
    ],
) -> Tuple[Optional[float], Optional[float], float]:
    if (
        detected_start is None
        or ln.visibility_start is None
        or ln.visibility_end is None
        or (float(ln.visibility_end) - float(ln.visibility_start)) < 8.0
    ):
        return detected_start, detected_end, detected_confidence

    hint = estimate_line_onset_hint_in_visibility_window(ln, group_frames, group_times)
    if hint is None or hint < detected_start + 2.5:
        return detected_start, detected_end, detected_confidence

    adjusted_start = hint
    adjusted_end = detected_end
    if adjusted_end is not None and adjusted_end < adjusted_start + 0.2:
        adjusted_end = None
    adjusted_conf = max(detected_confidence, 0.45)
    return adjusted_start, adjusted_end, adjusted_conf
