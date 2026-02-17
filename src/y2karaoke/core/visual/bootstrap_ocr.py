"""OCR frame sampling and caching helpers for visual bootstrap."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ...vision.ocr import get_ocr_engine

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency behavior
    cv2 = None  # type: ignore
    np = None  # type: ignore


def collect_raw_frames(
    video_path: Path,
    start: float,
    end: float,
    fps: float,
    roi_rect: tuple[int, int, int, int],
    *,
    log_fn: Any = None,
    ocr_engine_fn: Any = None,
) -> list[dict[str, Any]]:
    if cv2 is None or np is None:
        raise ImportError("OpenCV and Numpy are required.")

    if ocr_engine_fn is None:
        ocr_engine_fn = get_ocr_engine
    ocr = ocr_engine_fn()
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(src_fps / fps)), 1)
    rx, ry, rw, rh = roi_rect
    raw: list[dict[str, Any]] = []
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    frame_idx = max(int(round(start * src_fps)), 0)

    if log_fn:
        log_fn(f"Sampling frames at {fps} FPS...")

    while True:
        ok = cap.grab()
        if not ok:
            break
        t = frame_idx / src_fps
        if t > end + 0.2:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        ok, frame = cap.retrieve()
        if not ok:
            frame_idx += 1
            continue

        roi = frame[ry : ry + rh, rx : rx + rw]
        res = ocr.predict(roi)
        if res and res[0]:
            items = res[0]
            rec_texts = items.get("rec_texts", [])
            rec_boxes = items.get("rec_boxes", [])
            words = []
            for txt, box_data in zip(rec_texts, rec_boxes):
                points = box_data["word"] if isinstance(box_data, dict) else box_data
                nb = np.array(points).reshape(-1, 2)
                x, y = int(min(nb[:, 0])), int(min(nb[:, 1]))
                bw, bh = int(max(nb[:, 0]) - x), int(max(nb[:, 1]) - y)
                words.append({"text": txt, "x": x, "y": y, "w": bw, "h": bh})
            if words:
                raw.append({"time": t, "words": words})
        frame_idx += 1

    cap.release()
    return raw


def raw_frames_cache_path(
    video_path: Path,
    cache_dir: Path,
    fps: float,
    roi_rect: tuple[int, int, int, int],
    *,
    cache_version: str,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    sig = (
        f"{cache_version}:{video_path.resolve()}:{video_path.stat().st_mtime_ns}:"
        f"{video_path.stat().st_size}:{fps}:{roi_rect}"
    )
    digest = hashlib.md5(sig.encode()).hexdigest()
    return cache_dir / f"raw_frames_{digest}.json"


def collect_raw_frames_cached(
    video_path: Path,
    duration: float,
    fps: float,
    roi_rect: tuple[int, int, int, int],
    cache_dir: Path,
    *,
    cache_version: str,
    log_fn: Any = None,
    collect_fn: Any = None,
) -> list[dict[str, Any]]:
    cache_path = raw_frames_cache_path(
        video_path,
        cache_dir,
        fps,
        roi_rect,
        cache_version=cache_version,
    )
    if cache_path.exists():
        if log_fn:
            log_fn(f"Loading cached OCR frames: {cache_path.name}")
        return json.loads(cache_path.read_text())

    if collect_fn is None:
        collect_fn = collect_raw_frames
    raw = collect_fn(video_path, 0, duration, fps, roi_rect)
    cache_path.write_text(json.dumps(raw))
    return raw
