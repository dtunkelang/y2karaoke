"""Color analysis tools for karaoke video processing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, List, Any, cast

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore

from .ocr import get_ocr_engine, normalize_ocr_items
from ..exceptions import VisualRefinementError

logger = logging.getLogger(__name__)


def _foreground_mask(word_roi: np.ndarray) -> np.ndarray:
    """Estimate foreground text pixels inside a word box.

    OCR word boxes include substantial background area. A simple border-color
    model works well for karaoke overlays where text sits atop varying video
    backgrounds.
    """
    h, w = word_roi.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((0, 0), dtype=bool)

    edge = max(1, min(h, w) // 6)
    border = np.concatenate(
        [
            word_roi[:edge, :, :].reshape(-1, 3),
            word_roi[-edge:, :, :].reshape(-1, 3),
            word_roi[:, :edge, :].reshape(-1, 3),
            word_roi[:, -edge:, :].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)
    bg = np.median(border, axis=0)

    pixels = word_roi.reshape(-1, 3).astype(np.float32)
    dist_bg = np.linalg.norm(pixels - bg, axis=1)
    # Keep pixels that are sufficiently distinct from the local background.
    thresh = max(20.0, float(np.percentile(dist_bg, 72)))
    mask = (dist_bg >= thresh).reshape(h, w)

    if int(mask.sum()) < max(8, (h * w) // 40):
        # Fallback: keep bright/saturated pixels when contrast to border is weak.
        hsv = cv2.cvtColor(word_roi, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)
        mask = (sat >= 30) | (val >= 150)
    return mask


def _sample_text_pixels(word_roi: np.ndarray, max_points: int = 24) -> np.ndarray:
    mask = _foreground_mask(word_roi)
    if mask.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    pixels = word_roi[mask]
    if pixels.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if len(pixels) > max_points:
        idx = np.linspace(0, len(pixels) - 1, num=max_points, dtype=int)
        pixels = pixels[idx]
    return pixels.astype(np.float32)


def cluster_colors(pixel_samples: List[np.ndarray], k: int = 2) -> List[np.ndarray]:
    """Cluster a list of pixels into k colors using K-Means."""
    if not pixel_samples:
        return [np.array([255, 255, 255]), np.array([0, 0, 0])]

    if cv2 is None or np is None:
        raise ImportError("OpenCV and Numpy are required for color clustering.")

    data = np.stack(pixel_samples).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    # Explicitly pass None for labels to match cv2 overload
    _, _, centers = cv2.kmeans(data, k, None, criteria, 10, flags)  # type: ignore
    return [c for c in centers]


def infer_lyric_colors(
    video_path: Path, roi_rect: Tuple[int, int, int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample text colors from the video to find Unselected and Selected prototypes.
    Uses cluster support to identify stable colors vs transient fade-ins.

    Returns:
        Tuple of (c_unselected, c_selected, c_background_dummy)
    """
    if cv2 is None or np is None:
        raise ImportError("OpenCV and Numpy are required.")

    ocr = get_ocr_engine()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VisualRefinementError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / src_fps
    rx, ry, rw, rh = roi_rect
    pixel_samples: list[np.ndarray] = []

    # Sample middle 40% of the song where lyrics are dense
    start_t = duration * 0.3
    end_t = duration * 0.7
    step_t = 5.0

    logger.info(f"Inferring colors from {start_t:.1f}s to {end_t:.1f}s...")

    for t in np.arange(start_t, end_t, step_t):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        # Extract ROI
        if ry + rh > frame.shape[0] or rx + rw > frame.shape[1]:
            continue
        roi = frame[ry : ry + rh, rx : rx + rw]

        res = ocr.predict(roi)
        if res and res[0]:
            items = normalize_ocr_items(res[0])
            rec_boxes = items["rec_boxes"]

            for box_data in rec_boxes:
                points = box_data["word"] if isinstance(box_data, dict) else box_data
                nb = np.array(points).reshape(-1, 2)
                x, y = int(min(nb[:, 0])), int(min(nb[:, 1]))
                bw, bh = int(max(nb[:, 0]) - x), int(max(nb[:, 1]) - y)

                if bw > 4 and bh > 4:
                    y, x = max(0, y), max(0, x)
                    bh = min(bh, roi.shape[0] - y)
                    bw = min(bw, roi.shape[1] - x)
                    word_roi = roi[y : y + bh, x : x + bw]
                    sampled = _sample_text_pixels(word_roi)
                    if sampled.size:
                        pixel_samples.extend(sampled)

    cap.release()

    if not pixel_samples:
        logger.warning("No text detected for color inference. Using defaults.")
        return (
            np.array([255, 255, 255]),
            np.array([0, 0, 255]),
            np.array([0, 0, 0]),
        )

    # 1. Cluster into k=3 candidates
    data = np.stack(pixel_samples).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centers = cv2.kmeans(data, 3, cast(Any, None), criteria, 10, flags)

    # 2. Count support for each cluster
    counts = np.bincount(labels.flatten())
    cluster_info = []
    for i, center in enumerate(centers):
        cluster_info.append(
            {"center": center, "count": counts[i], "brightness": np.mean(center)}
        )

    # 3. Exclude the darkest cluster (likely background noise/edges)
    cluster_info.sort(key=lambda x: x["brightness"])
    foreground_clusters = cluster_info[1:]  # Keep two brightest

    # 4. Identify Selected vs Unselected
    # Heuristic:
    # - If Saturation difference is high, Colored one is Selected (active).
    # - Else (both grey or both colored), Brighter one is Selected (active).
    
    candidates = [c["center"] for c in foreground_clusters]
    if len(candidates) < 2:
        # Fallback: assume brighter is Unselected (standard white-to-blue)
        # But if we only have one, we make a dummy dark selected.
        c_un = candidates[0]
        c_sel = c_un * 0.6
    else:
        # Convert BGR to HSV for robust S/V comparison
        # centers are float32, need uint8 for cvtColor
        src = np.array([[c for c in candidates]], dtype=np.uint8)
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)[0]
        
        # hsv[i] = [H, S, V]
        s0, v0 = float(hsv[0][1]), float(hsv[0][2])
        s1, v1 = float(hsv[1][1]), float(hsv[1][2])
        
        diff_s = abs(s0 - s1)
        
        # If saturation differs significantly (> 15% of 255), prefer high saturation as Selected.
        if diff_s > 40:
            if s0 > s1:
                c_sel, c_un = candidates[0], candidates[1]
            else:
                c_sel, c_un = candidates[1], candidates[0]
        else:
            # Otherwise, prefer higher brightness as Selected (e.g. dim -> bright).
            # Note: OpenCV V channel is brightness.
            if v0 > v1:
                c_sel, c_un = candidates[0], candidates[1]
            else:
                c_sel, c_un = candidates[1], candidates[0]

    logger.info(
        f"Inferred stable colors: Unselected={c_un.astype(int)}, Selected={c_sel.astype(int)}"
    )
    return c_un, c_sel, cluster_info[0]["center"]


def classify_word_state(
    word_roi: np.ndarray, c_un: np.ndarray, c_sel: np.ndarray
) -> Tuple[str, float, float]:
    """
    Determine word state and return metrics for stability analysis.

    Returns:
        (state_str, ratio_of_selected_pixels, distance_to_unselected)
    """
    if cv2 is None or np is None:
        raise ImportError("OpenCV and Numpy required.")

    if word_roi.size == 0:
        return "unknown", 0.0, 1000.0

    mask = _foreground_mask(word_roi)
    pixels = word_roi[mask].astype(np.float32)
    if pixels.size == 0:
        pixels = word_roi.reshape(-1, 3).astype(np.float32)

    # Calculate distances to prototypes
    dist_un = np.linalg.norm(pixels - c_un, axis=1)
    dist_sel = np.linalg.norm(pixels - c_sel, axis=1)

    avg_dist_un = float(np.mean(dist_un))
    is_sel = dist_sel < dist_un
    ratio = float(np.mean(is_sel))

    if ratio > 0.8:
        state = "selected"
    elif ratio > 0.2:
        state = "mixed"
    else:
        state = "unselected"

    return state, ratio, avg_dist_un
