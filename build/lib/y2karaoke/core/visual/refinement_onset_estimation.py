"""Onset estimation helpers over full line visibility windows."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def estimate_onset_from_visibility_progress(
    vals: List[Dict[str, Any]],
) -> Tuple[Optional[float], float]:
    """Estimate onset from fractional rise within full visibility window."""
    if len(vals) < 10:
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

    deriv_onset, deriv_conf = estimate_onset_from_visibility_derivative(vals)

    if prog_onset is None:
        return deriv_onset, deriv_conf
    if deriv_onset is None:
        return prog_onset, prog_conf

    if deriv_conf >= (prog_conf - 0.08) and deriv_onset >= prog_onset + 0.15:
        return deriv_onset, max(prog_conf, deriv_conf)
    if deriv_conf > prog_conf + 0.12:
        return deriv_onset, deriv_conf
    return prog_onset, prog_conf


def estimate_onset_from_visibility_derivative(
    vals: List[Dict[str, Any]],
) -> Tuple[Optional[float], float]:
    """Estimate onset using first derivative of line activity over full visibility."""
    if len(vals) < 10:
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
