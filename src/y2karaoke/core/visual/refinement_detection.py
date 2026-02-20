"""Highlight detection helpers for visual timing refinement."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def detect_highlight_times(
    word_vals: List[Dict[str, Any]],
    *,
    detect_highlight_with_confidence: Callable[
        [List[Dict[str, Any]]], Tuple[Optional[float], Optional[float], float]
    ],
) -> Tuple[Optional[float], Optional[float]]:
    """Detect start and end times of visual highlight from color sequence."""
    s, e, _ = detect_highlight_with_confidence(word_vals)
    return s, e


def detect_highlight_with_confidence(
    word_vals: List[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[float], float]:
    """Detect highlight transition and estimate confidence in [0, 1]."""
    if len(word_vals) <= 10:
        return None, None, 0.0

    l_vals = np.array([v["avg"][0] for v in word_vals])
    kernel_size = min(10, len(l_vals))
    l_smooth = np.convolve(
        l_vals,
        np.ones(kernel_size) / kernel_size,
        mode="same",
    )

    idx_peak = int(np.argmax(l_smooth))
    c_initial = word_vals[idx_peak]["avg"]
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

    start_stable = max(0, idx_peak - 5)
    end_stable = min(len(dists_in), idx_peak + 5)
    stable_range = dists_in[start_stable:end_stable]

    # Very conservative noise floor for Karafun-style fades
    if stable_range:
        noise_floor = float(np.mean(stable_range) + 3 * np.std(stable_range))
    else:
        noise_floor = 2.5

    s, e = None, None
    for j in range(idx_peak, len(times)):
        # Require a sustained gradient rise to distinguish from slow fade-in
        if s is None and dists_in[j] > noise_floor:
            if j + 4 < len(times) and all(
                dists_in[j + k] > dists_in[j + k - 1] for k in range(1, 5)
            ):
                s = times[j]

        if s is not None and e is None:
            curr_dist_final = np.linalg.norm(word_vals[j]["avg"] - c_final)
            curr_dist_initial = np.linalg.norm(word_vals[j]["avg"] - c_initial)
            if curr_dist_final < curr_dist_initial:
                e = times[j]
                break

    strength = min(transition_norm / 25.0, 1.0)
    coverage = min(len(word_vals) / 40.0, 1.0)
    trigger_quality = 1.0 if (s is not None and e is not None and e >= s) else 0.35
    confidence = max(
        0.0, min(1.0, 0.5 * strength + 0.3 * coverage + 0.2 * trigger_quality)
    )
    return s, e, float(confidence)


def detect_sustained_onset(
    vals: List[Dict[str, Any]],
    *,
    min_start_time: Optional[float] = None,
) -> Tuple[Optional[float], float]:
    """Find first sustained color departure from baseline for line-level highlights."""
    if len(vals) < 6:
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


def detect_line_highlight_cycle(
    times: np.ndarray,
    activities: np.ndarray,
    present: np.ndarray,
    *,
    min_start_time: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float], float]:
    """Detect inactive->active->consumed cycle for a line highlight."""
    if len(times) < 8:
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
