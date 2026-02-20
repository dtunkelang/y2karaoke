"""Masking and line-color sampling helpers for visual refinement."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore

from ..models import TargetLine


def word_fill_mask(roi_bgr: np.ndarray, c_bg: np.ndarray) -> np.ndarray:
    """Create a mask for text pixels (foreground)."""
    if cv2 is None or np is None:
        raise ImportError("OpenCV and Numpy required.")

    dist_bg = np.linalg.norm(roi_bgr - c_bg, axis=2)
    mask = (dist_bg > 35).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    return mask


def line_fill_mask(roi_bgr: np.ndarray, c_bg: np.ndarray) -> np.ndarray:
    """Line-level text mask with lower contrast threshold for unselected text."""
    if cv2 is None or np is None:
        raise ImportError("OpenCV and Numpy required.")

    dist_bg = np.linalg.norm(roi_bgr - c_bg, axis=2)
    mask = (dist_bg > 15).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    return mask


def collect_line_color_values(
    ln: TargetLine,
    line_frames: List[Tuple[float, np.ndarray, np.ndarray]],
    c_bg_line: np.ndarray,
    *,
    line_fill_mask_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
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
        line_mask = line_fill_mask_fn(line_roi_bgr, c_bg_line)
        mask_count = int(np.sum(line_mask > 0))
        if mask_count <= 30:
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
