"""Visual timing refinement logic."""

from __future__ import annotations

import bisect
import logging
import math
from pathlib import Path
from typing import List, Optional, Any, Dict, Tuple, cast

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
    search_start = max(1, min(3, base_count // 2))
    start_idx = None
    for i in range(search_start, len(smooth) - hold + 1):
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
        mask_count = int(np.sum(line_mask > 0))
        if mask_count <= 30:
            # Keep visibility tracking alive with a relaxed foreground estimate.
            dist_bg = np.linalg.norm(line_roi_bgr - c_bg_line, axis=2)
            relaxed_mask = (dist_bg > 8).astype(np.uint8) * 255
            relaxed_count = int(np.sum(relaxed_mask > 0))
            if relaxed_count > 20:
                line_mask = relaxed_mask
                mask_count = relaxed_count
            else:
                line_mask = np.ones(line_roi_bgr.shape[:2], dtype=np.uint8) * 255
                mask_count = int(np.sum(line_mask > 0))
        line_roi_lab = roi_lab[y_lo:y_hi, x_lo:x_hi]
        vals.append(
            {
                "t": t,
                "mask": line_mask,
                "lab": line_roi_lab,
                "avg": line_roi_lab[line_mask.astype(bool)].mean(axis=0),
                "coverage": float(mask_count)
                / float(max(1, line_mask.shape[0] * line_mask.shape[1])),
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
    search_start = max(1, min(3, base_count // 2))
    start_idx = None
    for i in range(search_start, len(smooth) - hold + 1):
        seg = smooth[i : i + hold]
        if np.all(seg >= threshold) and seg[-1] >= seg[0]:
            start_idx = i
            break

    if start_idx is None:
        prog_onset: Optional[float] = None
        prog_conf = 0.0
    else:
        prog_onset = float(times[start_idx])
        prog_conf = max(0.0, min(1.0, dynamic / 80.0))

    deriv_onset, deriv_conf = _estimate_onset_from_visibility_derivative(vals)

    if prog_onset is None:
        return deriv_onset, deriv_conf
    if deriv_onset is None:
        return prog_onset, prog_conf

    # Prefer derivative onset when confidence is comparable and it is later;
    # this reduces appearance-as-onset false positives on long visibility windows.
    if deriv_conf >= (prog_conf - 0.08) and deriv_onset >= prog_onset + 0.15:
        return deriv_onset, max(prog_conf, deriv_conf)
    if deriv_conf > prog_conf + 0.12:
        return deriv_onset, deriv_conf
    return prog_onset, prog_conf


def _estimate_onset_from_visibility_derivative(
    vals: List[Dict[str, Any]],
) -> Tuple[Optional[float], float]:
    """Estimate onset using first derivative of line activity over full visibility."""
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

    deriv = np.diff(smooth, prepend=smooth[0])
    stable_deriv = deriv[:base_count]
    d_med = float(np.median(stable_deriv)) if len(stable_deriv) else 0.0
    d_mad = float(np.median(np.abs(stable_deriv - d_med))) if len(stable_deriv) else 0.0
    slope_threshold = max(0.45, d_med + max(0.3, 3.5 * d_mad))
    level_threshold = base_level + 0.15 * dynamic
    hold = max(2, min(4, len(smooth) // 12))
    search_start = max(1, min(3, base_count // 2))

    start_idx = None
    for i in range(search_start, len(smooth) - hold + 1):
        d_seg = deriv[i : i + hold]
        s_seg = smooth[i : i + hold]
        if np.all(d_seg >= slope_threshold) and np.all(s_seg >= level_threshold):
            start_idx = i
            break

    if start_idx is None:
        return None, 0.0

    slope_peak = float(np.max(deriv[start_idx:])) if start_idx < len(deriv) else 0.0
    slope_strength = (slope_peak - slope_threshold) / max(slope_threshold, 1e-6)
    confidence = max(
        0.0, min(1.0, 0.65 * min(dynamic / 80.0, 1.0) + 0.35 * slope_strength)
    )
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
        _assign_cluster_persistent_onsets(cluster, group_frames, group_times)


def _assign_cluster_persistent_onsets(
    cluster: List[TargetLine],
    group_frames: List[Tuple[float, np.ndarray, np.ndarray]],
    group_times: List[float],
) -> None:
    if len(cluster) < 3:
        return
    onset_candidates = _collect_persistent_block_onset_candidates(
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


def _line_has_assigned_word_timing(ln: TargetLine) -> bool:
    return bool(ln.word_starts and any(s is not None for s in ln.word_starts))


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


def _find_prev_end_floor(
    line_order: List[TargetLine], first_idx: int
) -> Optional[float]:
    for idx in range(first_idx - 1, -1, -1):
        prev = line_order[idx]
        if prev.word_ends and prev.word_ends[-1] is not None:
            return float(prev.word_ends[-1])
        if prev.end is not None:
            return float(prev.end)
    return None


def _find_next_start_cap(
    line_order: List[TargetLine],
    last_idx: int,
    prev_end_floor: Optional[float],
) -> Optional[float]:
    for idx in range(last_idx + 1, len(line_order)):
        nxt = line_order[idx]
        cand: Optional[float] = None
        if nxt.word_starts and nxt.word_starts[0] is not None:
            cand = float(nxt.word_starts[0])
        elif nxt.visibility_start is not None:
            cand = float(nxt.visibility_start)
        elif nxt.start is not None:
            cand = float(nxt.start)
        if cand is None:
            continue
        if prev_end_floor is None or cand > prev_end_floor + 0.2:
            return cand
    return None


def _assign_surrogate_cluster_timings(
    cluster: List[TargetLine],
    *,
    prev_end_floor: Optional[float],
    next_start_cap: Optional[float],
    onset_hints: Optional[Dict[int, float]] = None,
) -> None:
    block_start = max(
        min(float(ln.start) for ln in cluster),
        min(
            float(ln.visibility_start)
            for ln in cluster
            if ln.visibility_start is not None
        ),
    )
    if prev_end_floor is not None:
        block_start = max(block_start, prev_end_floor)

    block_end = (
        max(float(ln.visibility_end) for ln in cluster if ln.visibility_end is not None)
        + 1.0
    )
    if next_start_cap is not None:
        block_end = min(block_end, next_start_cap)

    n = len(cluster)
    min_line = 0.7
    line_gap = 0.2
    min_span = n * min_line + max(0, n - 1) * line_gap
    if block_end < block_start + min_span:
        block_end = block_start + min_span

    lead_slack = 0.0
    if not onset_hints and n == 2:
        long_vis = all(
            (ln.visibility_start is not None and ln.visibility_end is not None)
            and (float(ln.visibility_end) - float(ln.visibility_start)) >= 6.0
            for ln in cluster
        )
        if long_vis:
            max_slack = max(0.0, (block_end - block_start) - min_span)
            lead_slack = min(1.5, 0.6 * max_slack)

    span = block_end - (block_start + lead_slack)
    available = max(min_span, span) - max(0, n - 1) * line_gap
    weights = [math.sqrt(float(max(len(ln.words), 1))) for ln in cluster]
    w_sum = sum(weights) if sum(weights) > 0 else float(n)
    extra = max(0.0, available - n * min_line)
    durations = [min_line + extra * (w / w_sum) for w in weights]

    starts: List[float] = []
    cursor = block_start + lead_slack
    for dur in durations:
        starts.append(cursor)
        cursor += dur + line_gap

    if onset_hints:
        adjusted: List[float] = []
        prev_end = block_start - line_gap
        for i, (ln, dur) in enumerate(zip(cluster, durations)):
            s = starts[i]
            hint = onset_hints.get(id(ln))
            if hint is not None:
                s = max(s, float(hint))
            s = max(s, prev_end + line_gap)

            remaining = len(cluster) - i - 1
            min_tail = remaining * min_line + max(0, remaining) * line_gap
            max_s = block_end - min_tail - dur
            s = min(s, max_s)
            adjusted.append(s)
            prev_end = s + dur
        starts = adjusted

    for ln, line_start, dur in zip(cluster, starts, durations):
        line_end = line_start + dur
        _assign_line_level_word_timings(ln, line_start, line_end, 0.4)


def _collect_unresolved_line_onset_hints(
    cluster: List[TargetLine],
    group_frames: List[Tuple[float, np.ndarray, np.ndarray]],
    group_times: List[float],
) -> Dict[int, float]:
    hints: Dict[int, float] = {}
    for ln in cluster:
        if ln.visibility_start is None or ln.visibility_end is None:
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
            continue
        c_bg_line = np.mean(
            [
                np.mean(f[1], axis=(0, 1))
                for f in line_frames[: min(10, len(line_frames))]
            ],
            axis=0,
        )
        s, _e, conf = _detect_line_highlight_with_confidence(
            ln,
            line_frames,
            c_bg_line,
            min_start_time=float(ln.visibility_start),
            require_full_cycle=False,
        )
        if s is not None and conf >= 0.45:
            hints[id(ln)] = float(s)
            continue
        vals = _collect_line_color_values(ln, line_frames, c_bg_line)
        onset, onset_conf = _estimate_onset_from_visibility_progress(vals)
        if onset is not None and onset_conf >= 0.45:
            hints[id(ln)] = float(onset)
    return hints


def _estimate_line_onset_hint_in_visibility_window(
    ln: TargetLine,
    group_frames: List[Tuple[float, np.ndarray, np.ndarray]],
    group_times: List[float],
) -> Optional[float]:
    if ln.visibility_start is None or ln.visibility_end is None:
        return None
    v_start = max(0.0, float(ln.visibility_start) - 0.5)
    v_end = float(ln.visibility_end) + 0.5
    line_frames = _slice_frames_for_window(
        group_frames,
        group_times,
        v_start=v_start,
        v_end=v_end,
    )
    if len(line_frames) < 8:
        return None
    c_bg_line = np.mean(
        [np.mean(f[1], axis=(0, 1)) for f in line_frames[: min(10, len(line_frames))]],
        axis=0,
    )
    s, _e, conf = _detect_line_highlight_with_confidence(
        ln,
        line_frames,
        c_bg_line,
        min_start_time=float(ln.visibility_start),
        require_full_cycle=False,
    )
    if s is not None and conf >= 0.45:
        return float(s)
    vals = _collect_line_color_values(ln, line_frames, c_bg_line)
    onset, onset_conf = _estimate_onset_from_visibility_progress(vals)
    if onset is not None and onset_conf >= 0.45:
        return float(onset)
    return None


def _maybe_adjust_detected_line_start_with_visibility_hint(
    ln: TargetLine,
    *,
    detected_start: Optional[float],
    detected_end: Optional[float],
    detected_confidence: float,
    group_frames: List[Tuple[float, np.ndarray, np.ndarray]],
    group_times: List[float],
) -> Tuple[Optional[float], Optional[float], float]:
    if (
        detected_start is None
        or ln.visibility_start is None
        or ln.visibility_end is None
        or (float(ln.visibility_end) - float(ln.visibility_start)) < 8.0
    ):
        return detected_start, detected_end, detected_confidence

    hint = _estimate_line_onset_hint_in_visibility_window(ln, group_frames, group_times)
    if hint is None or hint < detected_start + 2.5:
        return detected_start, detected_end, detected_confidence

    adjusted_start = hint
    adjusted_end = detected_end
    if adjusted_end is not None and adjusted_end < adjusted_start + 0.2:
        adjusted_end = None
    adjusted_conf = max(detected_confidence, 0.45)
    return adjusted_start, adjusted_end, adjusted_conf


def _assign_surrogate_timings_for_unresolved_overlap_blocks(
    g_jobs: List[Tuple[TargetLine, float, float]],
    group_frames: Optional[List[Tuple[float, np.ndarray, np.ndarray]]] = None,
    group_times: Optional[List[float]] = None,
) -> None:
    unresolved = [
        ln
        for ln, _, _ in g_jobs
        if ln.visibility_start is not None
        and ln.visibility_end is not None
        and (float(ln.visibility_end) - float(ln.visibility_start)) >= 2.5
        and not _line_has_assigned_word_timing(ln)
    ]
    if len(unresolved) < 2:
        return

    clusters = _cluster_unresolved_visibility_lines(unresolved)
    line_order = [ln for ln, _, _ in g_jobs]
    line_pos = {id(ln): idx for idx, ln in enumerate(line_order)}
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        cluster.sort(key=lambda ln: (line_pos.get(id(ln), 10**9), float(ln.y)))

        first_idx = min(line_pos.get(id(ln), 10**9) for ln in cluster)
        last_idx = max(line_pos.get(id(ln), -1) for ln in cluster)
        prev_end_floor = _find_prev_end_floor(line_order, first_idx)
        next_start_cap = _find_next_start_cap(line_order, last_idx, prev_end_floor)
        onset_hints: Optional[Dict[int, float]] = None
        if group_frames is not None and group_times is not None:
            onset_hints = _collect_unresolved_line_onset_hints(
                cluster, group_frames, group_times
            )
        _assign_surrogate_cluster_timings(
            cluster,
            prev_end_floor=prev_end_floor,
            next_start_cap=next_start_cap,
            onset_hints=onset_hints,
        )


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
) -> Optional[float]:
    min_start: Optional[float] = None
    if ln.visibility_start is not None:
        min_start = float(ln.visibility_start)
        if ln.visibility_end is not None:
            vis_span = max(0.0, float(ln.visibility_end) - float(ln.visibility_start))
            # OCR visibility can lag true on-screen appearance; allow bounded lookback.
            if vis_span >= 8.0:
                min_start -= 2.0
            elif vis_span >= 5.0:
                min_start -= 1.0
        min_start = max(0.0, min_start)

    if last_assigned_start is not None:
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
            cursor += dur + line_gap


def _retime_large_gaps_with_early_visibility(  # noqa: C901
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    line_order = [ln for ln, _, _ in g_jobs]
    if len(line_order) < 2:
        return

    for idx in range(len(line_order)):
        curr = line_order[idx]
        curr_start = _line_start(curr)
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
            prev_end = _line_end(prev)
            if prev_end is None:
                continue
            prev_vis_end = (
                float(prev.visibility_end)
                if prev.visibility_end is not None
                else prev_end
            )
            delta = vis_start - prev_vis_end
            # Nearby visibility windows indicate sequential lyric flow.
            if -1.0 <= delta <= 2.5:
                anchor_end = prev_end
                break

        if anchor_end is None:
            continue
        gap = curr_start - anchor_end
        # Only retime local drift; skip long-range jumps where gaps are likely real.
        if gap > 6.0:
            continue
        if gap < 2.2:
            continue

        old_end = _line_end(curr)
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
            ns = _line_start(line_order[j])
            if ns is not None:
                next_start = ns
                break
        if next_start is not None:
            new_end = min(new_end, next_start - 0.15)
        new_end = max(new_end, new_start + 0.7)
        _assign_line_level_word_timings(curr, new_start, new_end, 0.45)


def _retime_followups_in_short_lead_shared_visibility_runs(  # noqa: C901
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
            first = run[0]
            second = run[1]
            first_start = _line_start(first)
            first_end = _line_end(first)
            second_start = _line_start(second)
            if (
                first_start is not None
                and first_end is not None
                and second_start is not None
            ):
                first_dur = max(0.0, first_end - first_start)
                if first_dur <= 1.1 and (second_start - first_end) <= 0.35:
                    last = run[-1]
                    last_end = _line_end(last)
                    if last_end is not None:
                        next_start: Optional[float] = None
                        for k in range(j, n):
                            ns = _line_start(line_order[k])
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
                                    s = _line_start(ln)
                                    e = _line_end(ln)
                                    if s is None or e is None:
                                        continue
                                    _assign_line_level_word_timings(
                                        ln,
                                        s + shift,
                                        e + shift,
                                        0.42,
                                    )

        i = j if j > i else i + 1


def _rebalance_two_followups_after_short_lead(
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    line_order = [ln for ln, _, _ in g_jobs]
    n = len(line_order)
    i = 0
    while i + 2 < n:
        a = line_order[i]
        b = line_order[i + 1]
        c = line_order[i + 2]
        if (
            a.visibility_start is None
            or a.visibility_end is None
            or b.visibility_start is None
            or b.visibility_end is None
            or c.visibility_start is None
            or c.visibility_end is None
        ):
            i += 1
            continue
        if not (
            abs(float(b.visibility_start) - float(a.visibility_start)) <= 1.0
            and abs(float(c.visibility_start) - float(a.visibility_start)) <= 1.0
            and abs(float(b.visibility_end) - float(a.visibility_end)) <= 2.0
            and abs(float(c.visibility_end) - float(a.visibility_end)) <= 2.0
        ):
            i += 1
            continue

        a_start = _line_start(a)
        a_end = _line_end(a)
        b_start = _line_start(b)
        c_start = _line_start(c)
        if a_start is None or a_end is None or b_start is None or c_start is None:
            i += 1
            continue
        if (a_end - a_start) > 1.15:
            i += 1
            continue
        if (c_start - b_start) >= 1.4:
            i += 1
            continue

        next_start: Optional[float] = None
        for k in range(i + 3, n):
            ns = _line_start(line_order[k])
            if ns is not None:
                next_start = ns
                break
        cap = (
            (next_start - 0.2)
            if next_start is not None
            else (float(a.visibility_end) + 0.8)
        )
        if cap <= b_start + 2.0:
            i += 1
            continue

        gap = 0.2
        avail = (cap - b_start) - gap
        if avail <= 1.4:
            i += 1
            continue
        min_line = 0.7
        extra = max(0.0, avail - 2.0 * min_line)
        w_b = max(1.0, math.sqrt(float(max(len(b.words), 1)))) * 1.8
        w_c = max(1.0, math.sqrt(float(max(len(c.words), 1))))
        w_sum = w_b + w_c
        dur_b = min_line + extra * (w_b / w_sum)
        dur_c = min_line + extra * (w_c / w_sum)
        s_b = b_start
        s_c = s_b + dur_b + gap
        _assign_line_level_word_timings(b, s_b, s_b + dur_b, 0.42)
        _assign_line_level_word_timings(c, s_c, s_c + dur_c, 0.42)
        i += 3


def _clamp_line_ends_to_visibility_windows(
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    line_order = [ln for ln, _, _ in g_jobs]
    for ln in line_order:
        if ln.visibility_start is None or ln.visibility_end is None:
            continue
        vis_span = float(ln.visibility_end) - float(ln.visibility_start)
        if vis_span < 4.0:
            continue
        s = _line_start(ln)
        e = _line_end(ln)
        if s is None or e is None:
            continue
        cap = float(ln.visibility_end) + 0.1
        if e <= cap + 1e-6:
            continue
        new_end = max(s + 0.7, cap)
        if new_end >= e - 0.05:
            continue
        _assign_line_level_word_timings(ln, s, new_end, 0.42)


def _pull_lines_earlier_after_visibility_transitions(
    g_jobs: List[Tuple[TargetLine, float, float]],
) -> None:
    line_order = [ln for ln, _, _ in g_jobs]
    for idx in range(1, len(line_order)):
        prev = line_order[idx - 1]
        curr = line_order[idx]
        prev_end = _line_end(prev)
        curr_start = _line_start(curr)
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

        old_end = _line_end(curr)
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
        _assign_line_level_word_timings(curr, new_start, new_end, 0.42)


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
        _assign_line_level_word_timings(
            b, b_sf - follower_shift, b_ef - follower_shift, 0.42
        )
        _assign_line_level_word_timings(
            c, c_sf - follower_shift, c_ef - follower_shift, 0.42
        )
        _assign_line_level_word_timings(
            d, d_sf - follower_shift, d_ef - follower_shift, 0.42
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

        shift = min(1.2, max(0.0, first_gap - 0.05))
        if shift < 0.35:
            continue

        _assign_line_level_word_timings(a, a_sf - shift, a_ef - shift, 0.42)
        _assign_line_level_word_timings(b, b_sf - shift, b_ef - shift, 0.42)
        _assign_line_level_word_timings(c, c_sf - shift, c_ef - shift, 0.42)
        _assign_line_level_word_timings(d, d_sf - shift, d_ef - shift, 0.42)


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
            line_min_start = _compute_line_min_start_time(
                ln,
                last_assigned_start=last_assigned_start,
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
            line_s, line_e, line_conf = (
                _maybe_adjust_detected_line_start_with_visibility_hint(
                    ln,
                    detected_start=line_s,
                    detected_end=line_e,
                    detected_confidence=line_conf,
                    group_frames=group_frames,
                    group_times=group_times,
                )
            )
            _assign_line_level_word_timings(ln, line_s, line_e, line_conf)
            if ln.word_starts and ln.word_starts[0] is not None:
                last_assigned_start = float(ln.word_starts[0])
            elif line_s is not None:
                last_assigned_start = float(line_s)

        _apply_persistent_block_highlight_order(g_jobs, group_frames, group_times)
        _assign_surrogate_timings_for_unresolved_overlap_blocks(
            g_jobs,
            group_frames=group_frames,
            group_times=group_times,
        )
        _retime_late_first_lines_in_shared_visibility_blocks(g_jobs)
        _retime_compressed_shared_visibility_blocks(g_jobs)
        for ln, _, _ in g_jobs:
            if ln.word_starts and ln.word_starts[0] is not None:
                if last_assigned_start is None:
                    last_assigned_start = float(ln.word_starts[0])
                else:
                    last_assigned_start = max(
                        last_assigned_start, float(ln.word_starts[0])
                    )

    _retime_large_gaps_with_early_visibility(jobs)
    _retime_followups_in_short_lead_shared_visibility_runs(jobs)
    _rebalance_two_followups_after_short_lead(jobs)
    _rebalance_early_lead_shared_visibility_runs(jobs)
    _shrink_overlong_leads_in_dense_shared_visibility_runs(jobs)
    _retime_dense_runs_after_overlong_lead(jobs)
    _pull_dense_short_runs_toward_previous_anchor(jobs)
    _clamp_line_ends_to_visibility_windows(jobs)
    _pull_lines_earlier_after_visibility_transitions(jobs)
    cap.release()
