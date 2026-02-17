"""Region of Interest (ROI) detection for karaoke lyrics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore

from .ocr import get_ocr_engine
from ..exceptions import VisualRefinementError

logger = logging.getLogger(__name__)


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

    all_boxes = []
    logger.info(f"Detecting ROI over {duration:.1f}s video...")

    # Sample 30s window in the middle of the song
    mid = duration / 2
    start_t = max(0, mid - 15)
    end_t = min(duration, mid + 15)
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

        res = ocr.predict(frame)
        if res and res[0]:
            items = res[0]
            # Handle list-of-dicts (Vision) vs list-of-lists (Paddle)
            # ocr.py standardizes to dict with 'rec_boxes'
            rec_boxes = items.get("rec_boxes", [])

            for box_data in rec_boxes:
                # box_data might be dict with 'word' key or just points
                if isinstance(box_data, dict):
                    points = box_data["word"]
                else:
                    points = box_data

                try:
                    nb = np.array(points).reshape(-1, 2)
                    x_min, y_min = np.min(nb, axis=0)
                    x_max, y_max = np.max(nb, axis=0)
                    all_boxes.append((x_min, y_min, x_max, y_max))
                except Exception:
                    continue
        frame_idx += 1

    cap.release()

    if not all_boxes:
        # Fallback to generous bottom-half ROI
        logger.warning("No text detected for ROI. Using default bottom-half.")
        return (
            int(width * 0.05),
            int(height * 0.4),
            int(width * 0.9),
            int(height * 0.5),
        )

    # Convert to numpy for easy stats
    boxes_np = np.array(all_boxes)

    # Find approximate bounds (10th/90th percentile to ignore outliers)
    # We want a union of all reasonable boxes
    x1 = int(np.percentile(boxes_np[:, 0], 5))
    y1 = int(np.percentile(boxes_np[:, 1], 5))
    x2 = int(np.percentile(boxes_np[:, 2], 95))
    y2 = int(np.percentile(boxes_np[:, 3], 95))

    # Add padding
    pad_x = int(width * 0.02)
    pad_y = int(height * 0.05)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    roi = (x1, y1, x2 - x1, y2 - y1)
    logger.info(f"Detected ROI: {roi}")
    return roi
