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


def _line_fill_mask(roi_bgr: np.ndarray, c_bg: np.ndarray) -> np.ndarray:
    """Line-level text mask with lower contrast threshold for unselected text."""
    if cv2 is None or np is None:
        raise ImportError("OpenCV and Numpy required.")

    dist_bg = np.linalg.norm(roi_bgr - c_bg, axis=2)
    mask = (dist_bg > 15).astype(np.uint8) * 255
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
        # Ensure refinement window is never shorter than observed visibility span.
        if ln.visibility_start is not None:
            line_start = min(line_start, float(ln.visibility_start))
        if ln.visibility_end is not None:
            line_end = max(line_end, float(ln.visibility_end))
        v_start, v_end = max(0.0, line_start - lead), line_end + tail
        jobs.append((ln, v_start, v_end))
    jobs.sort(key=lambda item: item[1])
    return jobs


def _detect_sustained_onset(
    vals: List[Dict[str, Any]],
    *,
    min_start_time: Optional[float] = None,
) -> Tuple[Optional[float], float]:
    """Find first sustained color departure from baseline for line-level highlights."""
    if np is None or len(vals) < 6:
        return None, 0.0

    times = np.array([float(v["t"]) for v in vals], dtype=np.float32)
    colors = np.array([v["avg"] for v in vals], dtype=np.float32)

    base_count = max(3, min(8, len(colors) // 3))
    baseline = colors[:base_count].mean(axis=0)
    activity = np.linalg.norm(colors - baseline, axis=1)

    kernel_size = min(5, len(activity))
    smooth = np.convolve(activity, np.ones(kernel_size) / kernel_size, mode="same")

    stable = smooth[:base_count]
    stable_med = float(np.median(stable)) if len(stable) else 0.0
    stable_mad = float(np.median(np.abs(stable - stable_med))) if len(stable) else 0.0
    threshold = max(1.8, stable_med + max(1.0, 3.5 * stable_mad))

    hold = max(2, min(4, len(smooth) // 8))
    start_idx = None
    for i in range(base_count, len(smooth) - hold + 1):
        t_i = float(times[i])
        if min_start_time is not None and t_i < min_start_time:
            continue
        segment = smooth[i : i + hold]
        if np.all(segment > threshold):
            start_idx = i
            break

    if start_idx is None:
        return None, 0.0

    peak = float(np.max(smooth[start_idx:])) if start_idx < len(smooth) else threshold
    confidence = max(0.0, min(1.0, (peak - threshold) / max(threshold, 1.0)))
    return float(times[start_idx]), confidence


def _detect_line_highlight_cycle(
    times: np.ndarray,
    activities: np.ndarray,
    present: np.ndarray,
    *,
    min_start_time: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float], float]:
    """Detect inactive->active->consumed cycle for a line highlight."""
    if np is None or len(times) < 8:
        return None, None, 0.0

    present_idx = np.where(present)[0]
    if len(present_idx) < 6:
        return None, None, 0.0

    base_count = max(3, min(8, len(present_idx) // 3))
    baseline_vals = activities[present_idx[:base_count]]
    stable_med = float(np.median(baseline_vals))
    stable_mad = float(np.median(np.abs(baseline_vals - stable_med)))
    start_threshold = max(1.8, stable_med + max(1.0, 3.5 * stable_mad))
    end_threshold = max(stable_med + 0.5, start_threshold * 0.55)

    hold_active = max(2, min(4, len(times) // 12))
    hold_inactive = max(2, min(5, len(times) // 10))

    start_idx = None
    for i in range(0, len(times) - hold_active + 1):
        if min_start_time is not None and float(times[i]) < min_start_time:
            continue
        if not np.all(present[i : i + hold_active]):
            continue
        if np.all(activities[i : i + hold_active] > start_threshold):
            start_idx = i
            break

    if start_idx is None:
        return None, None, 0.0

    end_idx = None
    for j in range(start_idx + hold_active, len(times) - hold_inactive + 1):
        if np.all(~present[j : j + hold_inactive]):
            end_idx = j
            break
        if np.all(present[j : j + hold_inactive]) and np.all(
            activities[j : j + hold_inactive] < end_threshold
        ):
            end_idx = j
            break

    if end_idx is None and np.sum(present[start_idx:]) <= hold_active + 1:
        end_idx = len(times) - 1

    peak = float(np.max(activities[start_idx:]))
    rise_strength = max(0.0, (peak - start_threshold) / max(start_threshold, 1.0))
    cycle_quality = 1.0 if end_idx is not None else 0.3
    confidence = max(0.0, min(1.0, 0.65 * rise_strength + 0.35 * cycle_quality))

    start_t = float(times[start_idx])
    end_t = float(times[end_idx]) if end_idx is not None else None
    if end_t is not None and end_t < start_t:
        end_t = None
    return start_t, end_t, confidence


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


def _read_window_frames_sampled(
    cap: Any,
    *,
    v_start: float,
    v_end: float,
    roi_rect: tuple[int, int, int, int],
    sample_fps: float,
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    if cv2 is None or np is None:
        raise ImportError("OpenCV and Numpy required.")
    if sample_fps <= 0:
        return _read_window_frames(cap, v_start=v_start, v_end=v_end, roi_rect=roi_rect)

    rx, ry, rw, rh = roi_rect
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(src_fps / max(sample_fps, 0.01))), 1)
    cap.set(cv2.CAP_PROP_POS_MSEC, v_start * 1000.0)
    frame_idx = max(int(round(v_start * src_fps)), 0)
    end_frame_idx = max(int(round(v_end * src_fps)), frame_idx)

    window_frames: List[Tuple[float, np.ndarray, np.ndarray]] = []
    while frame_idx <= end_frame_idx:
        ok = cap.grab()
        if not ok:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        ok, frame = cap.retrieve()
        if not ok:
            frame_idx += 1
            continue
        t = frame_idx / src_fps
        roi_bgr = frame[ry : ry + rh, rx : rx + rw]
        roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        window_frames.append((t, roi_bgr, roi_lab))
        frame_idx += 1

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


def _collect_line_color_values(
    ln: TargetLine,
    line_frames: List[Tuple[float, np.ndarray, np.ndarray]],
    c_bg_line: np.ndarray,
) -> List[Dict[str, Any]]:
    if cv2 is None or np is None:
        raise ImportError("OpenCV and Numpy required.")
    if not ln.word_rois:
        return []

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
        line_mask = _line_fill_mask(line_roi_bgr, c_bg_line)
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
    return vals


def _estimate_onset_from_visibility_progress(
    vals: List[Dict[str, Any]],
) -> Tuple[Optional[float], float]:
    """Estimate onset from fractional rise within full visibility window."""
    if np is None or len(vals) < 10:
        return None, 0.0

    times = np.array([float(v["t"]) for v in vals], dtype=np.float32)
    colors = np.array([v["avg"] for v in vals], dtype=np.float32)
    base_count = max(4, min(12, len(colors) // 4))
    baseline = colors[:base_count].mean(axis=0)
    activity = np.linalg.norm(colors - baseline, axis=1)
    kernel = min(5, len(activity))
    smooth = np.convolve(activity, np.ones(kernel) / kernel, mode="same")

    base_level = float(np.median(smooth[:base_count])) if base_count > 0 else 0.0
    peak = float(np.max(smooth))
    dynamic = peak - base_level
    if dynamic < 8.0:
        return None, 0.0

    threshold = base_level + 0.18 * dynamic
    hold = max(2, min(4, len(smooth) // 12))
    start_idx = None
    for i in range(base_count, len(smooth) - hold + 1):
        seg = smooth[i : i + hold]
        if np.all(seg >= threshold) and seg[-1] >= seg[0]:
            start_idx = i
            break

    if start_idx is None:
        return None, 0.0
    confidence = max(0.0, min(1.0, dynamic / 80.0))
    return float(times[start_idx]), confidence


def _apply_persistent_block_highlight_order(
    g_jobs: List[Tuple[TargetLine, float, float]],
    group_frames: List[Tuple[float, np.ndarray, np.ndarray]],
    group_times: List[float],
) -> None:
    """Override timings for persistent overlapping lines using highlight order."""
    candidates: List[TargetLine] = []
    for ln, _, _ in g_jobs:
        if not ln.word_rois or ln.visibility_start is None or ln.visibility_end is None:
            continue
        if (float(ln.visibility_end) - float(ln.visibility_start)) >= 12.0:
            candidates.append(ln)
    persistent = _select_persistent_overlap_lines(candidates)
    if len(persistent) < 3:
        return
    clusters = _cluster_persistent_lines_by_visibility(persistent)
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        onset_candidates = _collect_persistent_block_onset_candidates(
            cluster, group_frames, group_times
        )

        # Require at least two confident onsets to avoid overriding on weak evidence.
        if sum(1 for _, s, c in onset_candidates if s is not None and c >= 0.25) < 2:
            continue

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
            continue

        for i, (ln, onset, conf) in enumerate(assigned):
            next_onset = assigned[i + 1][1] if i + 1 < len(assigned) else None
            if next_onset is not None and (next_onset - onset) > 6.0:
                next_onset = None
            _assign_line_level_word_timings(
                ln,
                onset,
                next_onset,
                max(0.35, min(0.75, conf)),
            )


def _select_persistent_overlap_lines(candidates: List[TargetLine]) -> List[TargetLine]:
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


def _cluster_persistent_lines_by_visibility(
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


def _collect_persistent_block_onset_candidates(
    persistent: List[TargetLine],
    group_frames: List[Tuple[float, np.ndarray, np.ndarray]],
    group_times: List[float],
) -> List[Tuple[TargetLine, Optional[float], float]]:
    onset_candidates: List[Tuple[TargetLine, Optional[float], float]] = []
    for ln in persistent:
        if ln.visibility_start is None or ln.visibility_end is None:
            onset_candidates.append((ln, None, 0.0))
            continue
        v_start = max(0.0, float(ln.visibility_start) - 0.5)
        v_end = float(ln.visibility_end) + 0.5
        line_frames = _slice_frames_for_window(
            group_frames,
            group_times,
            v_start=v_start,
            v_end=v_end,
        )
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
        vals = _collect_line_color_values(ln, line_frames, c_bg_line)
        onset, conf = _estimate_onset_from_visibility_progress(vals)
        onset_candidates.append((ln, onset, conf))
    return onset_candidates


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
    *,
    min_start_time: Optional[float] = None,
    require_full_cycle: bool = False,
) -> Tuple[Optional[float], Optional[float], float]:
    if cv2 is None or np is None:
        raise ImportError("OpenCV and Numpy required.")
    if not ln.word_rois:
        return None, None, 0.0

    vals = _collect_line_color_values(ln, line_frames, c_bg_line)
    frame_times: List[float] = []
    frame_present: List[bool] = []
    frame_activity: List[float] = []
    baseline_samples: List[np.ndarray] = []

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

    cycle_s, cycle_e, cycle_conf = _detect_line_highlight_cycle(
        np.array(frame_times, dtype=np.float32),
        np.array(frame_activity, dtype=np.float32),
        np.array(frame_present, dtype=bool),
        min_start_time=min_start_time,
    )
    if cycle_s is not None and (cycle_e is not None or not require_full_cycle):
        return cycle_s, cycle_e, cycle_conf
    if require_full_cycle:
        return None, None, 0.0

    onset_s, onset_conf = _detect_sustained_onset(vals, min_start_time=min_start_time)
    generic_s, generic_e, generic_conf = _detect_highlight_with_confidence(vals)

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
    groups = _merge_line_refinement_jobs(jobs, max_group_duration_sec=90.0)

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


def refine_line_timings_at_low_fps(
    video_path: Path,
    target_lines: List[TargetLine],
    roi_rect: tuple[int, int, int, int],
    *,
    sample_fps: float = 6.0,
) -> None:
    """Refine line-level highlight timing cheaply when word-level cues are weak."""
    if cv2 is None or np is None:
        raise ImportError("OpenCV and Numpy required.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VisualRefinementError(f"Could not open video: {video_path}")

    logger.info(
        "Refining timings with low-FPS line-level highlight detection "
        f"(sample_fps={sample_fps:.1f})..."
    )
    jobs = _build_line_refinement_jobs(
        target_lines,
        lead_in_sec=10.0,
        tail_sec=1.0,
    )
    groups = _merge_line_refinement_jobs(jobs, max_group_duration_sec=90.0)
    last_assigned_start: Optional[float] = None

    for g_start, g_end, g_jobs in groups:
        group_frames = _read_window_frames_sampled(
            cap,
            v_start=g_start,
            v_end=g_end,
            roi_rect=roi_rect,
            sample_fps=sample_fps,
        )
        if len(group_frames) < 6:
            continue
        group_times = [frame[0] for frame in group_frames]

        for ln, v_start, v_end in g_jobs:
            line_frames = _slice_frames_for_window(
                group_frames,
                group_times,
                v_start=v_start,
                v_end=v_end,
            )
            if len(line_frames) < 6:
                continue
            line_min_start: Optional[float] = (
                float(ln.visibility_start) if ln.visibility_start is not None else None
            )
            if last_assigned_start is not None and (
                line_min_start is None or line_min_start >= last_assigned_start
            ):
                gate = float(last_assigned_start + 0.05)
                line_min_start = (
                    gate if line_min_start is None else max(line_min_start, gate)
                )
            c_bg_line = np.mean(
                [
                    np.mean(f[1], axis=(0, 1))
                    for f in line_frames[: min(10, len(line_frames))]
                ],
                axis=0,
            )
            line_s, line_e, line_conf = _detect_line_highlight_with_confidence(
                ln,
                line_frames,
                c_bg_line,
                min_start_time=line_min_start,
                require_full_cycle=True,
            )
            if line_s is None and line_e is None:
                continue
            _assign_line_level_word_timings(ln, line_s, line_e, line_conf)
            if ln.word_starts and ln.word_starts[0] is not None:
                last_assigned_start = float(ln.word_starts[0])
            elif line_s is not None:
                last_assigned_start = float(line_s)

        _apply_persistent_block_highlight_order(g_jobs, group_frames, group_times)
        for ln, _, _ in g_jobs:
            if ln.word_starts and ln.word_starts[0] is not None:
                if last_assigned_start is None:
                    last_assigned_start = float(ln.word_starts[0])
                else:
                    last_assigned_start = max(
                        last_assigned_start, float(ln.word_starts[0])
                    )

    cap.release()
