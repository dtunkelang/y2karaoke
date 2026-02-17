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

from .models import TargetLine
from ..exceptions import VisualRefinementError

logger = logging.getLogger(__name__)


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
    word_vals: List[Dict[str, Any]]
) -> Tuple[Optional[float], Optional[float]]:
    """Detect start and end times of visual highlight from color sequence."""
    if len(word_vals) <= 10:
        return None, None

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
        return None, None

    idx_valley = idx_peak + int(np.argmin(l_smooth[idx_peak:]))
    c_final = word_vals[idx_valley]["avg"]

    if np.linalg.norm(c_final - c_initial) <= 2.0:
        return None, None

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
    return s, e


def refine_word_timings_at_high_fps(  # noqa: C901
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

    rx, ry, rw, rh = roi_rect
    logger.info("Refining timings with Departure-Onset detection...")

    for i, ln in enumerate(target_lines):
        if not ln.word_rois:
            continue
        # Window: independent per line
        # Handle potential None end time by defaulting to start + 5s (safe upper bound)
        line_end = ln.end if ln.end is not None else ln.start + 5.0
        v_start, v_end = max(0.0, ln.start - 1.0), line_end + 1.0

        cap.set(cv2.CAP_PROP_POS_MSEC, v_start * 1000.0)
        line_frames = []
        while True:
            ok, frame = cap.read()
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not ok or t > v_end:
                break
            line_frames.append((t, frame[ry : ry + rh, rx : rx + rw]))

        if len(line_frames) < 20:
            continue

        # Estimate background color from first few frames (assumed unlit)
        c_bg_line = np.mean(
            [np.mean(f[1], axis=(0, 1)) for f in line_frames[:10]], axis=0
        )

        new_starts: List[Optional[float]] = []
        new_ends: List[Optional[float]] = []

        # We know word_rois is not None from check above
        assert ln.word_rois is not None

        for wi in range(len(ln.words)):
            wx, wy, ww, wh = ln.word_rois[wi]
            # 1. Identify TEXT-ONLY frames
            word_vals = []
            for t, roi in line_frames:
                if wy + wh <= roi.shape[0] and wx + ww <= roi.shape[1]:
                    word_roi = roi[wy : wy + wh, wx : wx + ww]
                    mask = _word_fill_mask(word_roi, c_bg_line)
                    if np.sum(mask > 0) > 30:  # Glyph is present
                        lab = cv2.cvtColor(word_roi, cv2.COLOR_BGR2LAB).astype(
                            np.float32
                        )
                        word_vals.append(
                            {
                                "t": t,
                                "mask": mask,
                                "lab": lab,
                                "avg": lab[mask.astype(bool)].mean(axis=0),
                            }
                        )

            s, e = _detect_highlight_times(word_vals)
            new_starts.append(s)
            new_ends.append(e)

        ln.word_starts = new_starts
        ln.word_ends = new_ends

    cap.release()

