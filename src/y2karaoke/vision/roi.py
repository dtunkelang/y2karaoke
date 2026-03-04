"""Region of Interest (ROI) detection for karaoke lyrics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore

from .ocr import get_ocr_engine, normalize_ocr_items
from ..exceptions import VisualRefinementError

logger = logging.getLogger(__name__)


def _default_bottom_half_roi(width: int, height: int) -> tuple[int, int, int, int]:
    return (
        int(width * 0.05),
        int(height * 0.4),
        int(width * 0.9),
        int(height * 0.5),
    )


def _extract_rec_boxes(prediction: object) -> list[object]:
    if not isinstance(prediction, list) or not prediction or not prediction[0]:
        return []
    items = normalize_ocr_items(prediction[0])
    return list(items["rec_boxes"])


def _box_points_array(box_data: object) -> Any | None:
    if np is None:
        return None
    points = box_data["word"] if isinstance(box_data, dict) else box_data
    try:
        return np.array(points).reshape(-1, 2)
    except Exception:
        return None


def _collect_boxes_from_window(
    *,
    cap: Any,
    ocr: Any,
    src_fps: float,
    sample_fps: float,
    start_t: float,
    end_t: float,
) -> list[tuple[float, float, float, float]]:
    if cv2 is None:
        return []
    all_boxes: list[tuple[float, float, float, float]] = []
    step = max(int(round(src_fps / max(sample_fps, 0.01))), 1)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_t * 1000.0)
    frame_idx = max(int(round(start_t * src_fps)), 0)
    end_frame_idx = max(int(round(end_t * src_fps)), frame_idx)

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
        for box_data in _extract_rec_boxes(ocr.predict(frame)):
            nb = _box_points_array(box_data)
            if nb is None:
                continue
            x_min, y_min = np.min(nb, axis=0)
            x_max, y_max = np.max(nb, axis=0)
            all_boxes.append((x_min, y_min, x_max, y_max))
        frame_idx += 1
    return all_boxes


def _apply_roi_guardrails(
    *,
    boxes_np: Any,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1 = int(np.percentile(boxes_np[:, 0], 2))
    y1 = int(np.percentile(boxes_np[:, 1], 3))
    x2 = int(np.percentile(boxes_np[:, 2], 98))
    y2 = int(np.percentile(boxes_np[:, 3], 97))

    pad_left = int(width * 0.06)
    pad_right = int(width * 0.04)
    pad_y = int(height * 0.05)

    x1 = max(0, x1 - pad_left)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_right)
    y2 = min(height, y2 + pad_y)

    min_w = int(width * 0.82)
    cur_w = max(1, x2 - x1)
    if cur_w < min_w:
        expand = (min_w - cur_w) // 2 + 1
        x1 = max(0, x1 - expand)
        x2 = min(width, x2 + expand)

    x1 = min(x1, int(width * 0.12))
    y1 = min(y1, int(height * 0.22))
    return x1, y1, x2 - x1, y2 - y1


def detect_lyric_roi(
    video_path: Path, sample_fps: float = 1.0
) -> Tuple[int, int, int, int]:
    """
    Detect the bounding box where lyrics appear most frequently.

    Returns:
        (x, y, width, height)
    """
    if cv2 is None or np is None:
        raise ImportError("OpenCV and Numpy required.")

    ocr = get_ocr_engine()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VisualRefinementError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / src_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Detecting ROI over {duration:.1f}s video...")

    # Sample 30s window in the middle of the song
    mid = duration / 2
    start_t = max(0, mid - 15)
    end_t = min(duration, mid + 15)
    all_boxes = _collect_boxes_from_window(
        cap=cap,
        ocr=ocr,
        src_fps=src_fps,
        sample_fps=sample_fps,
        start_t=start_t,
        end_t=end_t,
    )

    cap.release()

    if not all_boxes:
        # Fallback to generous bottom-half ROI
        logger.warning("No text detected for ROI. Using default bottom-half.")
        return _default_bottom_half_roi(width, height)

    boxes_np = np.array(all_boxes)
    roi = _apply_roi_guardrails(boxes_np=boxes_np, width=width, height=height)
    logger.info(f"Detected ROI: {roi}")
    return roi
