"""Visual timing refinement logic."""

from __future__ import annotations

import bisect
import logging
import math
from pathlib import Path
from typing import List, Optional, Any, Dict, Tuple

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore

from ..models import TargetLine
from ...exceptions import VisualRefinementError

logger = logging.getLogger(__name__)
_MAX_MERGED_WINDOW_SEC = 20.0


def _word_fill_mask(roi_bgr: np.ndarray, c_bg: np.ndarray) -> np.ndarray:
    """Create a mask for text pixels (foreground)."""
    if cv2 is None or np is None:
        raise ImportError("OpenCV and Numpy required.")

    dist_bg = np.linalg.norm(roi_bgr - c_bg, axis=2)
    mask = (dist_bg > 35).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    return mask


def _detect_highlight_times(
    word_vals: List[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[float]]:
    """Detect start and end times of visual highlight from color sequence."""
    s, e, _ = _detect_highlight_with_confidence(word_vals)
    return s, e


def _detect_highlight_with_confidence(
    word_vals: List[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[float], float]:
    """Detect highlight transition and estimate confidence in [0, 1]."""
    if len(word_vals) <= 10:
        return None, None, 0.0

    l_vals = np.array([v["avg"][0] for v in word_vals])
    # Smooth lightness curve
    kernel_size = min(10, len(l_vals))
    l_smooth = np.convolve(
        l_vals,
        np.ones(kernel_size) / kernel_size,
        mode="same",
    )

    idx_peak = int(np.argmax(l_smooth))
    c_initial = word_vals[idx_peak]["avg"]

    # Find valley after peak
    if idx_peak >= len(l_smooth) - 1:
        return None, None, 0.0

    idx_valley = idx_peak + int(np.argmin(l_smooth[idx_peak:]))
    c_final = word_vals[idx_valley]["avg"]

    transition_norm = float(np.linalg.norm(c_final - c_initial))
    if transition_norm <= 2.0:
        return None, None, 0.0

    times = []
    dists_in = []
    for v in word_vals:
        times.append(v["t"])
        dists_in.append(np.linalg.norm(v["avg"] - c_initial))

    # 2. Departure search: find exact frame where color starts moving
    # Calculate noise floor from stable period around peak
    start_stable = max(0, idx_peak - 5)
    end_stable = min(len(dists_in), idx_peak + 5)
    stable_range = dists_in[start_stable:end_stable]

    noise_floor = 1.0
    if stable_range:
        noise_floor = float(np.mean(stable_range) + 2 * np.std(stable_range))

    s, e = None, None
    for j in range(idx_peak, len(times)):
        # Start trigger: Consistent departure from noise floor
        if s is None and dists_in[j] > noise_floor:
            if j + 3 < len(times) and all(
                dists_in[j + k] > dists_in[j + k - 1] for k in range(1, 4)
            ):
                s = times[j]

        # End trigger: Closer to final state
        if s is not None and e is None:
            curr_dist_final = np.linalg.norm(word_vals[j]["avg"] - c_final)
            curr_dist_initial = np.linalg.norm(word_vals[j]["avg"] - c_initial)
            if curr_dist_final < curr_dist_initial:
                e = times[j]
                break

    # Confidence combines transition strength, sample coverage, and trigger quality.
    strength = min(transition_norm / 25.0, 1.0)
    coverage = min(len(word_vals) / 40.0, 1.0)
    trigger_quality = 1.0 if (s is not None and e is not None and e >= s) else 0.35
    confidence = max(
        0.0, min(1.0, 0.5 * strength + 0.3 * coverage + 0.2 * trigger_quality)
    )
    return s, e, float(confidence)


def _build_line_refinement_jobs(
    target_lines: List[TargetLine],
) -> List[Tuple[TargetLine, float, float]]:
    jobs: List[Tuple[TargetLine, float, float]] = []
    for ln in target_lines:
        if not ln.word_rois:
            continue
        line_end = ln.end if ln.end is not None else ln.start + 5.0
        v_start, v_end = max(0.0, ln.start - 1.0), line_end + 1.0
        jobs.append((ln, v_start, v_end))
    jobs.sort(key=lambda item: item[1])
    return jobs


def _merge_line_refinement_jobs(
    jobs: List[Tuple[TargetLine, float, float]],
    *,
    max_group_duration_sec: float = _MAX_MERGED_WINDOW_SEC,
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


def _read_window_frames(
    cap: Any,
    *,
    v_start: float,
    v_end: float,
    roi_rect: tuple[int, int, int, int],
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    rx, ry, rw, rh = roi_rect
    cap.set(cv2.CAP_PROP_POS_MSEC, v_start * 1000.0)
    window_frames = []
    while True:
        ok, frame = cap.read()
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if not ok or t > v_end:
            break
        roi_bgr = frame[ry : ry + rh, rx : rx + rw]
        roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        window_frames.append((t, roi_bgr, roi_lab))
    return window_frames


def _slice_frames_for_window(
    group_frames: List[Tuple[float, np.ndarray, np.ndarray]],
    group_times: List[float],
    *,
    v_start: float,
    v_end: float,
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    lo = bisect.bisect_left(group_times, v_start)
    hi = bisect.bisect_right(group_times, v_end)
    return group_frames[lo:hi]


def _refine_line_with_frames(
    ln: TargetLine,
    line_frames: List[Tuple[float, np.ndarray, np.ndarray]],
) -> None:
    # Estimate background color from first few frames (assumed unlit)
    c_bg_line = np.mean([np.mean(f[1], axis=(0, 1)) for f in line_frames[:10]], axis=0)

    new_starts: List[Optional[float]] = []
    new_ends: List[Optional[float]] = []
    new_confidences: List[Optional[float]] = []

    # We know word_rois is not None from job construction
    assert ln.word_rois is not None

    for wi in range(len(ln.words)):
        wx, wy, ww, wh = ln.word_rois[wi]
        word_vals = []
        for t, roi, roi_lab in line_frames:
            if wy + wh <= roi.shape[0] and wx + ww <= roi.shape[1]:
                word_roi = roi[wy : wy + wh, wx : wx + ww]
                mask = _word_fill_mask(word_roi, c_bg_line)
                if np.sum(mask > 0) > 30:  # Glyph is present
                    lab = roi_lab[wy : wy + wh, wx : wx + ww]
                    word_vals.append(
                        {
                            "t": t,
                            "mask": mask,
                            "lab": lab,
                            "avg": lab[mask.astype(bool)].mean(axis=0),
                        }
                    )

        s, e, conf = _detect_highlight_with_confidence(word_vals)
        new_starts.append(s)
        new_ends.append(e)
        new_confidences.append(conf)

    # For line-level karaoke (all words change together), word-level transitions may
    # be absent. Fall back to line transition + weighted in-line distribution.
    if not any(s is not None for s in new_starts):
        line_s, line_e, line_conf = _detect_line_highlight_with_confidence(
            ln, line_frames, c_bg_line
        )
        _assign_line_level_word_timings(ln, line_s, line_e, line_conf)
        return

    ln.word_starts = new_starts
    ln.word_ends = new_ends
    ln.word_confidences = new_confidences


def _detect_line_highlight_with_confidence(
    ln: TargetLine,
    line_frames: List[Tuple[float, np.ndarray, np.ndarray]],
    c_bg_line: np.ndarray,
) -> Tuple[Optional[float], Optional[float], float]:
    if cv2 is None or np is None:
        raise ImportError("OpenCV and Numpy required.")
    if not ln.word_rois:
        return None, None, 0.0

    x1 = min(wx for wx, _, _, _ in ln.word_rois)
    y1 = min(wy for _, wy, _, _ in ln.word_rois)
    x2 = max(wx + ww for wx, _, ww, _ in ln.word_rois)
    y2 = max(wy + wh for _, wy, _, wh in ln.word_rois)

    vals: List[Dict[str, Any]] = []
    for t, roi_bgr, roi_lab in line_frames:
        x_lo = max(0, x1)
        y_lo = max(0, y1)
        x_hi = min(roi_bgr.shape[1], x2)
        y_hi = min(roi_bgr.shape[0], y2)
        if x_hi <= x_lo or y_hi <= y_lo:
            continue

        line_roi_bgr = roi_bgr[y_lo:y_hi, x_lo:x_hi]
        line_mask = _word_fill_mask(line_roi_bgr, c_bg_line)
        if np.sum(line_mask > 0) <= 30:
            continue

        line_roi_lab = roi_lab[y_lo:y_hi, x_lo:x_hi]
        vals.append(
            {
                "t": t,
                "mask": line_mask,
                "lab": line_roi_lab,
                "avg": line_roi_lab[line_mask.astype(bool)].mean(axis=0),
            }
        )

    return _detect_highlight_with_confidence(vals)


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


def refine_word_timings_at_high_fps(
    video_path: Path,
    target_lines: List[TargetLine],
    roi_rect: tuple[int, int, int, int],
) -> None:
    """Refine start/end times for words in target_lines using high-FPS analysis."""
    if cv2 is None or np is None:
        raise ImportError("OpenCV and Numpy required.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VisualRefinementError(f"Could not open video: {video_path}")

    logger.info("Refining timings with Departure-Onset detection...")
    jobs = _build_line_refinement_jobs(target_lines)
    groups = _merge_line_refinement_jobs(jobs)

    for g_start, g_end, g_jobs in groups:
        group_frames = _read_window_frames(
            cap,
            v_start=g_start,
            v_end=g_end,
            roi_rect=roi_rect,
        )
        if not group_frames:
            continue
        group_times = [frame[0] for frame in group_frames]

        for ln, v_start, v_end in g_jobs:
            line_frames = _slice_frames_for_window(
                group_frames,
                group_times,
                v_start=v_start,
                v_end=v_end,
            )
            if len(line_frames) < 20:
                continue
            _refine_line_with_frames(ln, line_frames)

    cap.release()
