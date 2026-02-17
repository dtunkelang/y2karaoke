"""OCR frame sampling and caching helpers for visual bootstrap."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from ...vision.ocr import get_ocr_engine, normalize_ocr_items

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency behavior
    cv2 = None  # type: ignore
    np = None  # type: ignore


def _predict_rois_with_fallback(
    ocr: Any,
    rois: list[Any],
    supports_batch: bool | None,
) -> tuple[list[Any], bool | None]:
    if not rois:
        return [], supports_batch

    if supports_batch is not False:
        try:
            pred = ocr.predict(rois)
            if isinstance(pred, list) and len(pred) == len(rois):
                return pred, True
        except Exception:
            supports_batch = False

    out: list[Any] = []
    for roi_nd in rois:
        try:
            single = ocr.predict(roi_nd)
        except Exception:
            out.append(None)
            continue
        if single and isinstance(single, list):
            out.append(single[0])
        else:
            out.append(None)
    return out, supports_batch


def _append_predicted_words(
    raw: list[dict[str, Any]],
    pred_items: list[Any],
    times: list[float],
    *,
    roi_shapes: list[tuple[int, int]] | None = None,
) -> None:
    if np is None:
        raise ImportError("Numpy is required.")

    for idx, (t_val, raw_item) in enumerate(zip(times, pred_items)):
        if not raw_item:
            continue
        roi_h = roi_w = None
        if roi_shapes is not None and idx < len(roi_shapes):
            roi_h, roi_w = roi_shapes[idx]
        items = normalize_ocr_items(raw_item)
        rec_texts = items["rec_texts"]
        rec_boxes = items["rec_boxes"]
        words = []
        for txt, box_data in zip(rec_texts, rec_boxes):
            points = box_data["word"] if isinstance(box_data, dict) else box_data
            nb = np.array(points).reshape(-1, 2)
            x, y = int(min(nb[:, 0])), int(min(nb[:, 1]))
            bw, bh = int(max(nb[:, 0]) - x), int(max(nb[:, 1]) - y)
            if bw <= 0 or bh <= 0:
                continue
            if roi_h is not None and roi_w is not None:
                if x < -2 or y < -2:
                    continue
                if x > int(roi_w * 1.05) or y > int(roi_h * 1.05):
                    continue
            words.append({"text": txt, "x": x, "y": y, "w": bw, "h": bh})
        if words:
            raw.append({"time": t_val, "words": words})


def _overlay_root(token: str) -> str:
    compact = "".join(ch for ch in token.lower() if ch.isalnum())
    return compact[:4]


def _is_edge_position(x: float, y: float, *, roi_width: int, roi_height: int) -> bool:
    return x >= roi_width * 0.75 or y >= roi_height * 0.86


def _collect_edge_overlay_stats(
    raw_frames: list[dict[str, Any]],
    *,
    roi_width: int,
    roi_height: int,
) -> tuple[
    dict[tuple[str, int, int], dict[str, float]],
    dict[str, int],
    dict[str, set[str]],
]:
    bins: dict[tuple[str, int, int], dict[str, float]] = {}
    root_edge_frame_counts: dict[str, int] = {}
    root_variants: dict[str, set[str]] = {}
    for frame in raw_frames:
        seen_edge_roots: set[str] = set()
        for w in frame.get("words", []):
            if not isinstance(w, dict):
                continue
            text = str(w.get("text", ""))
            compact = re.sub(r"[^a-z0-9]", "", text.lower())
            if len(compact) < 4:
                continue
            root = _overlay_root(compact)
            if not root:
                continue
            x = float(w.get("x", 0.0))
            y = float(w.get("y", 0.0))
            if not _is_edge_position(x, y, roi_width=roi_width, roi_height=roi_height):
                continue
            root_variants.setdefault(root, set()).add(compact)
            seen_edge_roots.add(root)
            key = (root, int(round(x / 24.0)), int(round(y / 24.0)))
            rec = bins.setdefault(
                key,
                {
                    "count": 0.0,
                    "sum_x": 0.0,
                    "sum_y": 0.0,
                    "sum_x2": 0.0,
                    "sum_y2": 0.0,
                },
            )
            rec["count"] += 1.0
            rec["sum_x"] += x
            rec["sum_y"] += y
            rec["sum_x2"] += x * x
            rec["sum_y2"] += y * y
        for root in seen_edge_roots:
            root_edge_frame_counts[root] = root_edge_frame_counts.get(root, 0) + 1
    return bins, root_edge_frame_counts, root_variants


def _identify_banned_overlay(
    bins: dict[tuple[str, int, int], dict[str, float]],
    root_edge_frame_counts: dict[str, int],
    root_variants: dict[str, set[str]],
    *,
    total_frames: int,
    roi_width: int,
    roi_height: int,
) -> tuple[set[tuple[str, int, int]], set[str]]:
    banned_keys: set[tuple[str, int, int]] = set()
    for key, rec in bins.items():
        root = key[0]
        n = max(rec["count"], 1.0)
        freq = rec["count"] / float(total_frames)
        mean_x = rec["sum_x"] / n
        mean_y = rec["sum_y"] / n
        var_x = max(rec["sum_x2"] / n - mean_x * mean_x, 0.0)
        var_y = max(rec["sum_y2"] / n - mean_y * mean_y, 0.0)
        if (
            freq >= 0.22
            and (var_x**0.5) <= 18.0
            and (var_y**0.5) <= 18.0
            and _is_edge_position(
                mean_x, mean_y, roi_width=roi_width, roi_height=roi_height
            )
            and len(root_variants.get(root, set())) >= 2
        ):
            banned_keys.add(key)

    banned_roots = {
        root
        for root, count in root_edge_frame_counts.items()
        if (count / float(total_frames)) >= 0.25
        and len(root_variants.get(root, set())) >= 2
    }
    return banned_keys, banned_roots


def _filter_banned_overlay_words(
    raw_frames: list[dict[str, Any]],
    *,
    roi_width: int,
    roi_height: int,
    banned_keys: set[tuple[str, int, int]],
    banned_roots: set[str],
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for frame in raw_frames:
        new_words = []
        for w in frame.get("words", []):
            if not isinstance(w, dict):
                continue
            text = str(w.get("text", ""))
            compact = re.sub(r"[^a-z0-9]", "", text.lower())
            root = _overlay_root(compact)
            x = float(w.get("x", 0.0))
            y = float(w.get("y", 0.0))
            key = (root, int(round(x / 24.0)), int(round(y / 24.0)))
            is_edge = _is_edge_position(
                x, y, roi_width=roi_width, roi_height=roi_height
            )
            is_short_extreme_edge = (
                bool(compact)
                and len(compact) <= 4
                and (x >= roi_width * 0.9 or y >= roi_height * 0.9)
            )
            if banned_roots and is_short_extreme_edge:
                continue
            if root and is_edge and (key in banned_keys or root in banned_roots):
                continue
            new_words.append(w)
        filtered.append({**frame, "words": new_words})
    return filtered


def _suppress_persistent_edge_overlay_words(
    raw_frames: list[dict[str, Any]],
    *,
    roi_width: int,
    roi_height: int,
) -> list[dict[str, Any]]:
    if len(raw_frames) < 20:
        return raw_frames
    if roi_width <= 0 or roi_height <= 0:
        return raw_frames

    total_frames = len(raw_frames)
    bins, root_edge_frame_counts, root_variants = _collect_edge_overlay_stats(
        raw_frames, roi_width=roi_width, roi_height=roi_height
    )
    banned_keys, banned_roots = _identify_banned_overlay(
        bins,
        root_edge_frame_counts,
        root_variants,
        total_frames=total_frames,
        roi_width=roi_width,
        roi_height=roi_height,
    )
    if not banned_keys and not banned_roots:
        return raw_frames

    return _filter_banned_overlay_words(
        raw_frames,
        roi_width=roi_width,
        roi_height=roi_height,
        banned_keys=banned_keys,
        banned_roots=banned_roots,
    )


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

    batch_size = 8
    supports_batch: bool | None = None
    buffered_rois: list[Any] = []
    buffered_shapes: list[tuple[int, int]] = []
    buffered_times: list[float] = []

    def _flush_batch() -> None:
        nonlocal supports_batch
        if not buffered_rois:
            return
        pred_items, supports_batch = _predict_rois_with_fallback(
            ocr, buffered_rois, supports_batch
        )
        _append_predicted_words(
            raw,
            pred_items,
            buffered_times,
            roi_shapes=buffered_shapes,
        )
        buffered_rois.clear()
        buffered_shapes.clear()
        buffered_times.clear()

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
        buffered_rois.append(roi)
        buffered_shapes.append((roi.shape[0], roi.shape[1]))
        buffered_times.append(t)
        if len(buffered_rois) >= batch_size:
            _flush_batch()
        frame_idx += 1

    _flush_batch()
    cap.release()
    return _suppress_persistent_edge_overlay_words(raw, roi_width=rw, roi_height=rh)


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
        loaded = json.loads(cache_path.read_text())
        return _suppress_persistent_edge_overlay_words(
            loaded, roi_width=roi_rect[2], roi_height=roi_rect[3]
        )

    if collect_fn is None:
        collect_fn = collect_raw_frames
    raw = collect_fn(video_path, 0, duration, fps, roi_rect)
    raw = _suppress_persistent_edge_overlay_words(
        raw, roi_width=roi_rect[2], roi_height=roi_rect[3]
    )
    cache_path.write_text(json.dumps(raw))
    return raw
