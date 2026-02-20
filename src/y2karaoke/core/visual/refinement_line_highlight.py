"""Line-level highlight detection helper."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from ..models import TargetLine

FrameTriplet = Tuple[float, Any, Any]


def detect_line_highlight_with_confidence(
    ln: TargetLine,
    line_frames: List[FrameTriplet],
    c_bg_line: Any,
    *,
    min_start_time: Optional[float] = None,
    require_full_cycle: bool = False,
    collect_line_color_values: Callable[
        [TargetLine, List[FrameTriplet], Any], List[Dict[str, Any]]
    ],
    detect_line_highlight_cycle: Callable[
        [Any, Any, Any, Optional[float]],
        Tuple[Optional[float], Optional[float], float],
    ],
    detect_sustained_onset: Callable[
        [List[Dict[str, Any]], Optional[float]], Tuple[Optional[float], float]
    ],
    detect_highlight_with_confidence: Callable[
        [List[Dict[str, Any]]], Tuple[Optional[float], Optional[float], float]
    ],
) -> Tuple[Optional[float], Optional[float], float]:
    if np is None:
        raise ImportError("Numpy required.")
    if not ln.word_rois:
        return None, None, 0.0

    vals = collect_line_color_values(ln, line_frames, c_bg_line)
    frame_times: List[float] = []
    frame_present: List[bool] = []
    frame_activity: List[float] = []
    baseline_samples: List[Any] = []

    val_by_time = {float(v["t"]): v for v in vals}
    for t, _roi_bgr, _roi_lab in line_frames:
        frame_times.append(float(t))
        rec = val_by_time.get(float(t))
        if rec is None:
            frame_present.append(False)
            frame_activity.append(0.0)
            continue

        avg = rec["avg"]
        frame_present.append(True)
        if len(baseline_samples) < 8:
            baseline_samples.append(avg)
        baseline = (
            np.mean(np.array(baseline_samples, dtype=np.float32), axis=0)
            if baseline_samples
            else avg
        )
        frame_activity.append(float(np.linalg.norm(avg - baseline)))

    if not vals:
        return None, None, 0.0

    cycle_s, cycle_e, cycle_conf = detect_line_highlight_cycle(
        np.array(frame_times, dtype=np.float32),
        np.array(frame_activity, dtype=np.float32),
        np.array(frame_present, dtype=bool),
        min_start_time,
    )
    if cycle_s is not None and (cycle_e is not None or not require_full_cycle):
        return cycle_s, cycle_e, cycle_conf
    if require_full_cycle:
        return None, None, 0.0

    onset_s, onset_conf = detect_sustained_onset(vals, min_start_time)
    generic_s, generic_e, generic_conf = detect_highlight_with_confidence(vals)

    if onset_s is not None and onset_conf >= 0.15:
        end = generic_e
        if end is not None and end < onset_s:
            end = None
        confidence = max(onset_conf, 0.6 * generic_conf)
        return onset_s, end, float(max(0.0, min(1.0, confidence)))

    if generic_s is not None:
        if min_start_time is not None and generic_s < min_start_time:
            return None, None, 0.0
        return generic_s, generic_e, generic_conf

    return None, None, 0.0
