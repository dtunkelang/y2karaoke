"""Shared frame-window IO/slicing helpers for visual refinement."""

from __future__ import annotations

import bisect
from typing import Any, List, Tuple

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore


def read_window_frames(
    cap: Any,
    *,
    v_start: float,
    v_end: float,
    roi_rect: tuple[int, int, int, int],
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    if cv2 is None or np is None:
        raise ImportError("OpenCV and Numpy required.")

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


def read_window_frames_sampled(
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
        return read_window_frames(cap, v_start=v_start, v_end=v_end, roi_rect=roi_rect)

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


def slice_frames_for_window(
    group_frames: List[Tuple[float, np.ndarray, np.ndarray]],
    group_times: List[float],
    *,
    v_start: float,
    v_end: float,
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    lo = bisect.bisect_left(group_times, v_start)
    hi = bisect.bisect_right(group_times, v_end)
    return group_frames[lo:hi]
