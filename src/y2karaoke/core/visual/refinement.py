"""Visual timing refinement logic."""

from __future__ import annotations

import logging
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
from .surrogate_timing import (
    _assign_surrogate_cluster_timings as _assign_surrogate_cluster_timings_impl,
    _assign_surrogate_timings_for_unresolved_overlap_blocks as _assign_surrogate_timings_for_unresolved_overlap_blocks_impl,
)
from .refinement_postpasses import (
    _clamp_line_ends_to_visibility_windows,
    _compress_overlong_sparse_line_timings,
    _compute_line_min_start_time,
    _pull_dense_short_runs_toward_previous_anchor,
    _pull_lines_earlier_after_visibility_transitions,
    _rebalance_early_lead_shared_visibility_runs,
    _rebalance_two_followups_after_short_lead,
    _promote_unresolved_first_repeated_lines,
    _retime_compressed_shared_visibility_blocks,
    _retime_dense_runs_after_overlong_lead,
    _retime_followups_in_short_lead_shared_visibility_runs,
    _retime_large_gaps_with_early_visibility,
    _retime_late_first_lines_in_shared_visibility_blocks,
    _rebalance_middle_lines_in_four_line_shared_visibility_runs,
    _pull_late_first_lines_in_alternating_repeated_blocks,
    _retime_repeated_blocks_with_long_tail_gap,
    _retime_short_interstitial_lines_between_anchors,
    _shrink_overlong_leads_in_dense_shared_visibility_runs,
)
from .refinement_frame_windows import (
    read_window_frames as _read_window_frames_impl,
)
from .refinement_frame_windows import (
    read_window_frames_sampled as _read_window_frames_sampled_impl,
)
from .refinement_frame_windows import (
    slice_frames_for_window as _slice_frames_for_window_impl,
)
from .refinement_persistent_blocks import (
    apply_persistent_block_highlight_order as _apply_persistent_block_highlight_order_impl,
)
from .refinement_persistent_blocks import (
    assign_cluster_persistent_onsets as _assign_cluster_persistent_onsets_impl,
)
from .refinement_persistent_blocks import (
    cluster_persistent_lines_by_visibility as _cluster_persistent_lines_by_visibility_impl,
)
from .refinement_persistent_blocks import (
    collect_persistent_block_onset_candidates as _collect_persistent_block_onset_candidates_impl,
)
from .refinement_persistent_blocks import (
    select_persistent_overlap_lines as _select_persistent_overlap_lines_impl,
)
from .refinement_overlap_hints import (
    collect_unresolved_line_onset_hints as _collect_unresolved_line_onset_hints_impl,
)
from .refinement_overlap_hints import (
    estimate_line_onset_hint_in_visibility_window as _estimate_line_onset_hint_in_visibility_window_impl,
)
from .refinement_overlap_hints import (
    line_has_assigned_word_timing as _line_has_assigned_word_timing_impl,
)
from .refinement_overlap_hints import (
    maybe_adjust_detected_line_start_with_visibility_hint as _maybe_adjust_detected_line_start_with_visibility_hint_impl,
)
from .refinement_detection import (
    detect_highlight_times as _detect_highlight_times_impl,
)
from .refinement_detection import (
    detect_highlight_with_confidence as _detect_highlight_with_confidence_impl,
)
from .refinement_detection import (
    detect_line_highlight_cycle as _detect_line_highlight_cycle_impl,
)
from .refinement_detection import (
    detect_sustained_onset as _detect_sustained_onset_impl,
)
from .refinement_jobs import (
    build_line_refinement_jobs as _build_line_refinement_jobs_impl,
)
from .refinement_jobs import (
    merge_line_refinement_jobs as _merge_line_refinement_jobs_impl,
)
from .refinement_line_assignment import (
    assign_line_level_word_timings as _assign_line_level_word_timings_impl,
)

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
    return _detect_highlight_times_impl(
        word_vals,
        detect_highlight_with_confidence=_detect_highlight_with_confidence,
    )


def _detect_highlight_with_confidence(
    word_vals: List[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[float], float]:
    return _detect_highlight_with_confidence_impl(word_vals)


def _build_line_refinement_jobs(
    target_lines: List[TargetLine],
    *,
    lead_in_sec: float = 1.0,
    tail_sec: float = 1.0,
) -> List[Tuple[TargetLine, float, float]]:
    return _build_line_refinement_jobs_impl(
        target_lines,
        lead_in_sec=lead_in_sec,
        tail_sec=tail_sec,
    )


def _detect_sustained_onset(
    vals: List[Dict[str, Any]],
    *,
    min_start_time: Optional[float] = None,
) -> Tuple[Optional[float], float]:
    return _detect_sustained_onset_impl(vals, min_start_time=min_start_time)


def _detect_line_highlight_cycle(
    times: np.ndarray,
    activities: np.ndarray,
    present: np.ndarray,
    *,
    min_start_time: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float], float]:
    return _detect_line_highlight_cycle_impl(
        times,
        activities,
        present,
        min_start_time=min_start_time,
    )


def _merge_line_refinement_jobs(
    jobs: List[Tuple[TargetLine, float, float]],
    *,
    max_group_duration_sec: float = _MAX_MERGED_WINDOW_SEC,
) -> List[Tuple[float, float, List[Tuple[TargetLine, float, float]]]]:
    return _merge_line_refinement_jobs_impl(
        jobs,
        max_group_duration_sec=max_group_duration_sec,
    )


def _read_window_frames(
    cap: Any,
    *,
    v_start: float,
    v_end: float,
    roi_rect: tuple[int, int, int, int],
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    return _read_window_frames_impl(
        cap,
        v_start=v_start,
        v_end=v_end,
        roi_rect=roi_rect,
    )


def _read_window_frames_sampled(
    cap: Any,
    *,
    v_start: float,
    v_end: float,
    roi_rect: tuple[int, int, int, int],
    sample_fps: float,
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    return _read_window_frames_sampled_impl(
        cap,
        v_start=v_start,
        v_end=v_end,
        roi_rect=roi_rect,
        sample_fps=sample_fps,
    )


def _slice_frames_for_window(
    group_frames: List[Tuple[float, np.ndarray, np.ndarray]],
    group_times: List[float],
    *,
    v_start: float,
    v_end: float,
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    return _slice_frames_for_window_impl(
        group_frames,
        group_times,
        v_start=v_start,
        v_end=v_end,
    )


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
    _apply_persistent_block_highlight_order_impl(
        g_jobs,
        group_frames,
        group_times,
        select_persistent_overlap_lines=_select_persistent_overlap_lines,
        cluster_persistent_lines_by_visibility=_cluster_persistent_lines_by_visibility,
        assign_cluster_persistent_onsets=_assign_cluster_persistent_onsets,
    )


def _assign_cluster_persistent_onsets(
    cluster: List[TargetLine],
    group_frames: List[Tuple[float, np.ndarray, np.ndarray]],
    group_times: List[float],
) -> None:
    _assign_cluster_persistent_onsets_impl(
        cluster,
        group_frames,
        group_times,
        collect_persistent_block_onset_candidates=_collect_persistent_block_onset_candidates,
        assign_line_level_word_timings=_assign_line_level_word_timings,
    )


def _select_persistent_overlap_lines(candidates: List[TargetLine]) -> List[TargetLine]:
    return _select_persistent_overlap_lines_impl(candidates)


def _cluster_persistent_lines_by_visibility(
    lines: List[TargetLine],
) -> List[List[TargetLine]]:
    return _cluster_persistent_lines_by_visibility_impl(lines)


def _collect_persistent_block_onset_candidates(
    persistent: List[TargetLine],
    group_frames: List[Tuple[float, np.ndarray, np.ndarray]],
    group_times: List[float],
) -> List[Tuple[TargetLine, Optional[float], float]]:
    return _collect_persistent_block_onset_candidates_impl(
        persistent,
        group_frames,
        group_times,
        slice_frames_for_window=lambda frames, times, start, end: _slice_frames_for_window(
            frames, times, v_start=start, v_end=end
        ),
        collect_line_color_values=_collect_line_color_values,
        estimate_onset_from_visibility_progress=_estimate_onset_from_visibility_progress,
    )


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
        min_start = (
            float(ln.visibility_start) if ln.visibility_start is not None else None
        )
        line_s, line_e, line_conf = _detect_line_highlight_with_confidence(
            ln,
            line_frames,
            c_bg_line,
            min_start_time=min_start,
            require_full_cycle=True,
        )
        if line_s is None and line_e is None:
            line_s, line_e, line_conf = _detect_line_highlight_with_confidence(
                ln,
                line_frames,
                c_bg_line,
                min_start_time=min_start,
                require_full_cycle=False,
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
    _assign_line_level_word_timings_impl(
        ln,
        line_start,
        line_end,
        line_confidence,
    )


def _line_has_assigned_word_timing(ln: TargetLine) -> bool:
    return _line_has_assigned_word_timing_impl(ln)


def _detect_line_highlight_for_overlap_hint(
    ln: TargetLine,
    line_frames: List[Tuple[float, np.ndarray, np.ndarray]],
    c_bg_line: np.ndarray,
    min_start: float,
    require_cycle: bool,
) -> Tuple[Optional[float], Optional[float], float]:
    return _detect_line_highlight_with_confidence(
        ln,
        line_frames,
        c_bg_line,
        min_start_time=min_start,
        require_full_cycle=require_cycle,
    )


def _assign_surrogate_cluster_timings(
    cluster: List[TargetLine],
    *,
    prev_end_floor: Optional[float],
    next_start_cap: Optional[float],
    onset_hints: Optional[Dict[int, float]] = None,
) -> None:
    _assign_surrogate_cluster_timings_impl(
        cluster,
        prev_end_floor=prev_end_floor,
        next_start_cap=next_start_cap,
        onset_hints=onset_hints,
        assign_line_level_word_timings_fn=_assign_line_level_word_timings,
    )


def _collect_unresolved_line_onset_hints(
    cluster: List[TargetLine],
    group_frames: List[Tuple[float, np.ndarray, np.ndarray]],
    group_times: List[float],
) -> Dict[int, float]:
    return _collect_unresolved_line_onset_hints_impl(
        cluster,
        group_frames,
        group_times,
        slice_frames_for_window=lambda frames, times, start, end: _slice_frames_for_window(
            frames,
            times,
            v_start=start,
            v_end=end,
        ),
        detect_line_highlight_with_confidence=_detect_line_highlight_for_overlap_hint,
        collect_line_color_values=_collect_line_color_values,
        estimate_onset_from_visibility_progress=_estimate_onset_from_visibility_progress,
    )


def _estimate_line_onset_hint_in_visibility_window(
    ln: TargetLine,
    group_frames: List[Tuple[float, np.ndarray, np.ndarray]],
    group_times: List[float],
) -> Optional[float]:
    return _estimate_line_onset_hint_in_visibility_window_impl(
        ln,
        group_frames,
        group_times,
        slice_frames_for_window=lambda frames, times, start, end: _slice_frames_for_window(
            frames,
            times,
            v_start=start,
            v_end=end,
        ),
        detect_line_highlight_with_confidence=_detect_line_highlight_for_overlap_hint,
        collect_line_color_values=_collect_line_color_values,
        estimate_onset_from_visibility_progress=_estimate_onset_from_visibility_progress,
    )


def _maybe_adjust_detected_line_start_with_visibility_hint(
    ln: TargetLine,
    *,
    detected_start: Optional[float],
    detected_end: Optional[float],
    detected_confidence: float,
    group_frames: List[Tuple[float, np.ndarray, np.ndarray]],
    group_times: List[float],
) -> Tuple[Optional[float], Optional[float], float]:
    return _maybe_adjust_detected_line_start_with_visibility_hint_impl(
        ln,
        detected_start=detected_start,
        detected_end=detected_end,
        detected_confidence=detected_confidence,
        group_frames=group_frames,
        group_times=group_times,
        estimate_line_onset_hint_in_visibility_window=lambda ln, frames, times: _estimate_line_onset_hint_in_visibility_window(
            ln,
            frames,
            times,
        ),
    )


def _assign_surrogate_timings_for_unresolved_overlap_blocks(
    g_jobs: List[Tuple[TargetLine, float, float]],
    group_frames: Optional[List[Tuple[float, np.ndarray, np.ndarray]]] = None,
    group_times: Optional[List[float]] = None,
) -> None:
    hint_collector = None
    if group_frames is not None and group_times is not None:

        def _hint_collector(cluster: List[TargetLine]) -> Dict[int, float]:
            return _collect_unresolved_line_onset_hints(
                cluster, group_frames, group_times
            )

        hint_collector = _hint_collector

    _assign_surrogate_timings_for_unresolved_overlap_blocks_impl(
        g_jobs,
        line_has_assigned_word_timing_fn=_line_has_assigned_word_timing,
        assign_surrogate_cluster_timings_fn=_assign_surrogate_cluster_timings,
        collect_onset_hints_fn=hint_collector,
    )


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

        _apply_persistent_block_highlight_order(g_jobs, group_frames, group_times)
        _assign_surrogate_timings_for_unresolved_overlap_blocks(
            g_jobs,
            group_frames=group_frames,
            group_times=group_times,
        )
        _retime_late_first_lines_in_shared_visibility_blocks(g_jobs)
        _retime_compressed_shared_visibility_blocks(g_jobs)
        _promote_unresolved_first_repeated_lines(g_jobs)
        _compress_overlong_sparse_line_timings(g_jobs)

    _retime_large_gaps_with_early_visibility(jobs)
    _retime_followups_in_short_lead_shared_visibility_runs(jobs)
    _rebalance_two_followups_after_short_lead(jobs)
    _rebalance_early_lead_shared_visibility_runs(jobs)
    _shrink_overlong_leads_in_dense_shared_visibility_runs(jobs)
    _retime_dense_runs_after_overlong_lead(jobs)
    _pull_dense_short_runs_toward_previous_anchor(jobs)
    _retime_repeated_blocks_with_long_tail_gap(jobs)
    _pull_late_first_lines_in_alternating_repeated_blocks(jobs)
    _clamp_line_ends_to_visibility_windows(jobs)
    _pull_lines_earlier_after_visibility_transitions(jobs)
    _retime_short_interstitial_lines_between_anchors(jobs)
    _rebalance_middle_lines_in_four_line_shared_visibility_runs(jobs)
    cap.release()


def refine_line_timings_at_low_fps(  # noqa: C901
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
    last_assigned_visibility_end: Optional[float] = None

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
                last_assigned_visibility_end=last_assigned_visibility_end,
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
                # inside the sampled window. Fall back to onset-only detection.
                line_s, line_e, line_conf = _detect_line_highlight_with_confidence(
                    ln,
                    line_frames,
                    c_bg_line,
                    min_start_time=line_min_start,
                    require_full_cycle=False,
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
                if ln.visibility_end is not None:
                    last_assigned_visibility_end = float(ln.visibility_end)
            elif line_s is not None:
                last_assigned_start = float(line_s)
                if ln.visibility_end is not None:
                    last_assigned_visibility_end = float(ln.visibility_end)

        _apply_persistent_block_highlight_order(g_jobs, group_frames, group_times)
        _assign_surrogate_timings_for_unresolved_overlap_blocks(
            g_jobs,
            group_frames=group_frames,
            group_times=group_times,
        )
        _retime_late_first_lines_in_shared_visibility_blocks(g_jobs)
        _retime_compressed_shared_visibility_blocks(g_jobs)
        _promote_unresolved_first_repeated_lines(g_jobs)
        _compress_overlong_sparse_line_timings(g_jobs)
        for ln, _, _ in g_jobs:
            if ln.word_starts and ln.word_starts[0] is not None:
                if last_assigned_start is None:
                    last_assigned_start = float(ln.word_starts[0])
                else:
                    last_assigned_start = max(
                        last_assigned_start, float(ln.word_starts[0])
                    )
                if ln.visibility_end is not None:
                    vis_end = float(ln.visibility_end)
                    if last_assigned_visibility_end is None:
                        last_assigned_visibility_end = vis_end
                    else:
                        last_assigned_visibility_end = max(
                            last_assigned_visibility_end, vis_end
                        )

    _retime_large_gaps_with_early_visibility(jobs)
    _retime_followups_in_short_lead_shared_visibility_runs(jobs)
    _rebalance_two_followups_after_short_lead(jobs)
    _rebalance_early_lead_shared_visibility_runs(jobs)
    _shrink_overlong_leads_in_dense_shared_visibility_runs(jobs)
    _retime_dense_runs_after_overlong_lead(jobs)
    _pull_dense_short_runs_toward_previous_anchor(jobs)
    _retime_repeated_blocks_with_long_tail_gap(jobs)
    _pull_late_first_lines_in_alternating_repeated_blocks(jobs)
    _clamp_line_ends_to_visibility_windows(jobs)
    _pull_lines_earlier_after_visibility_transitions(jobs)
    _retime_short_interstitial_lines_between_anchors(jobs)
    _rebalance_middle_lines_in_four_line_shared_visibility_runs(jobs)
    cap.release()
