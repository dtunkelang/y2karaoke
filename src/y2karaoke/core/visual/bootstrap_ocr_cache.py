"""Caching helpers for OCR bootstrap raw frame collection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def raw_frames_cache_path(
    video_path: Path,
    cache_dir: Path,
    fps: float,
    roi_rect: tuple[int, int, int, int],
    *,
    cache_version: str,
    ocr_fingerprint_fn: Any,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ocr_fingerprint = ocr_fingerprint_fn()
    sig = (
        f"{cache_version}:{video_path.resolve()}:{video_path.stat().st_mtime_ns}:"
        f"{video_path.stat().st_size}:{fps}:{roi_rect}:{ocr_fingerprint}"
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
    log_fn: Any,
    collect_fn: Any,
    raw_frames_cache_path_fn: Any,
    collect_raw_frames_fn: Any,
    apply_post_ocr_filters_fn: Any,
) -> list[dict[str, Any]]:
    cache_path = raw_frames_cache_path_fn(
        video_path,
        cache_dir,
        fps,
        roi_rect,
        cache_version=cache_version,
    )
    if cache_path.exists():
        if log_fn:
            log_fn(f"Loading cached OCR frames: {cache_path.name}")
        loaded = json.loads(cache_path.read_text())
        return apply_post_ocr_filters_fn(
            loaded, roi_width=roi_rect[2], roi_height=roi_rect[3]
        )

    if collect_fn is None:
        raw = collect_raw_frames_fn(
            video_path,
            0,
            duration,
            fps,
            roi_rect,
            apply_post_filters=False,
        )
    else:
        raw = collect_fn(video_path, 0, duration, fps, roi_rect)
    raw = apply_post_ocr_filters_fn(raw, roi_width=roi_rect[2], roi_height=roi_rect[3])
    cache_path.write_text(json.dumps(raw))
    return raw
