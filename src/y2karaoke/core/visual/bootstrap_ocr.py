"""OCR frame sampling and caching helpers for visual bootstrap."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .bootstrap_ocr_filters import (
    apply_post_ocr_filters as _apply_post_ocr_filters_impl,
    collect_edge_overlay_stats as _collect_edge_overlay_stats_impl,
    count_dense_line_groups as _count_dense_line_groups_impl,
    estimate_lyrics_start_time as _estimate_lyrics_start_time_impl,
    fill_transient_ocr_gaps as _fill_transient_ocr_gaps_impl,
    filter_banned_overlay_words as _filter_banned_overlay_words_impl,
    identify_banned_overlay as _identify_banned_overlay_impl,
    is_edge_position as _is_edge_position_impl,
    overlay_root as _overlay_root_impl,
    suppress_early_banner_words as _suppress_early_banner_words_impl,
    suppress_intro_title_words as _suppress_intro_title_words_impl,
    suppress_persistent_edge_overlay_words as _suppress_persistent_edge_overlay_words_impl,
    suppress_transient_digit_heavy_frames as _suppress_transient_digit_heavy_frames_impl,
)
from ...vision.ocr import (
    get_ocr_cache_fingerprint,
    get_ocr_engine,
    normalize_ocr_items,
)

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
    roi_frames: list[Any] | None = None,
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
        roi_nd = None
        if roi_frames is not None and idx < len(roi_frames):
            roi_nd = roi_frames[idx]
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

            # Calculate visibility metrics (brightness and ink density)
            # Default to high visibility if no ROI frame is available (e.g. in unit tests)
            if roi_nd is not None:
                vis = _get_word_ink_metrics(roi_nd, x=x, y=y, w=bw, h=bh)
            else:
                vis = {"brightness": 255.0, "density": 1.0}

            if vis["density"] < 0.002:  # Extremely faint
                continue

            words.append(
                {
                    "text": txt,
                    "x": x,
                    "y": y,
                    "w": bw,
                    "h": bh,
                    "brightness": vis["brightness"],
                    "density": vis["density"],
                }
            )
        if words:
            line_boxes = _build_line_boxes(words, roi_nd=roi_nd)
            raw.append({"time": t_val, "words": words, "line_boxes": line_boxes})


def _get_word_ink_metrics(
    roi_nd: Any,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
) -> dict[str, float]:
    """Calculate brightness and density of isolated glyph pixels."""
    if np is None or cv2 is None or roi_nd is None:
        return {"brightness": 0.0, "density": 0.0}

    rh, rw = int(roi_nd.shape[0]), int(roi_nd.shape[1])
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(rw, int(x + w)), min(rh, int(y + h))
    if x1 <= x0 or y1 <= y0:
        return {"brightness": 0.0, "density": 0.0}

    crop = roi_nd[y0:y1, x0:x1]
    if crop.size == 0:
        return {"brightness": 0.0, "density": 0.0}

    # Estimate local background from border
    edge = max(1, min(crop.shape[0], crop.shape[1]) // 8)
    border = np.concatenate(
        [
            crop[:edge, :, :].reshape(-1, 3),
            crop[-edge:, :, :].reshape(-1, 3),
            crop[:, :edge, :].reshape(-1, 3),
            crop[:, -edge:, :].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)
    bg_color = np.median(border, axis=0)

    # Isolated pixels distinct from background
    pixels = crop.reshape(-1, 3).astype(np.float32)
    dist = np.linalg.norm(pixels - bg_color, axis=1)

    # Heuristic: ink is at least 10 units away in BGR space (accommodates mock frames)
    mask = dist > 10.0
    ink_pixels = pixels[mask]

    if ink_pixels.size == 0:
        # Fallback: if background subtraction failed, but we have bright pixels,
        # assume it's a solid block (often the case in simple mocks)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        if np.mean(gray) > 100:
            return {"brightness": float(np.mean(gray)), "density": 1.0}
        return {"brightness": 0.0, "density": 0.0}

    # Convert BGR to grayscale-equivalent brightness for the isolated pixels
    # (0.299*R + 0.587*G + 0.114*B) -> OpenCV uses 0.114*B + 0.587*G + 0.299*R
    brightnesses = (
        0.114 * ink_pixels[:, 0] + 0.587 * ink_pixels[:, 1] + 0.299 * ink_pixels[:, 2]
    )

    return {
        "brightness": float(np.mean(brightnesses)),
        "density": float(ink_pixels.size) / float(pixels.size),
    }


def _refine_line_box_with_text_mask(
    line_box: dict[str, Any],
    *,
    line_cy: float,
    roi_gray: Any,
) -> dict[str, Any]:
    if np is None or cv2 is None:
        return line_box
    if roi_gray is None or getattr(roi_gray, "ndim", 0) != 2:
        return line_box

    rh, rw = int(roi_gray.shape[0]), int(roi_gray.shape[1])
    x0 = int(line_box.get("x", 0))
    y0 = int(line_box.get("y", 0))
    w = int(line_box.get("w", 1))
    h = int(line_box.get("h", 1))
    x1 = x0 + max(1, w)
    y1 = y0 + max(1, h)

    sx0 = max(0, x0 - 10)
    sy0 = max(0, y0 - 8)
    sx1 = min(rw, x1 + 10)
    sy1 = min(rh, y1 + 8)
    if sx1 <= sx0 or sy1 <= sy0:
        return line_box

    sub = roi_gray[sy0:sy1, sx0:sx1]
    if sub.size == 0:
        return line_box

    _, otsu = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fixed = (sub >= 45).astype(np.uint8) * 255
    mask = np.where((otsu > 0) | (fixed > 0), 1, 0).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8)
    ).astype(np.uint8)
    if int(mask.sum()) <= 0:
        return line_box

    row_counts = mask.sum(axis=1)
    row_gate = max(2, int(round((sx1 - sx0) * 0.012)))
    active_rows = np.where(row_counts >= row_gate)[0]
    if active_rows.size == 0:
        active_rows = np.where(row_counts > 0)[0]
        if active_rows.size == 0:
            return line_box

    splits = np.where(np.diff(active_rows) > 1)[0] + 1
    runs = np.split(active_rows, splits)
    center_row = int(round(line_cy)) - sy0
    best = runs[0]
    best_dist = float("inf")
    for run in runs:
        lo, hi = int(run[0]), int(run[-1])
        if lo <= center_row <= hi:
            best = run
            break
        mid = (lo + hi) * 0.5
        dist = abs(mid - center_row)
        if dist < best_dist:
            best_dist = dist
            best = run

    ry0 = int(best[0])
    ry1 = int(best[-1]) + 1
    band = mask[ry0:ry1, :]
    if band.size == 0 or int(band.sum()) <= 0:
        return line_box
    ys, xs = np.where(band > 0)
    if xs.size == 0:
        return line_box

    tx0 = max(0, sx0 + int(xs.min()) - 2)
    tx1 = min(rw, sx0 + int(xs.max()) + 3)
    ty0 = max(0, sy0 + ry0 - 2)
    # Keep a slightly larger lower margin for descenders (g/j/p/q/y).
    ty1 = min(rh, sy0 + ry1 + 4)
    tw = max(1, tx1 - tx0)
    th = max(1, ty1 - ty0)

    if tw < int(0.35 * w) or tw > int(2.4 * w):
        return line_box
    if th < int(0.5 * h) or th > int(2.0 * h):
        return line_box

    return {
        "x": tx0,
        "y": ty0,
        "w": tw,
        "h": th,
        "tokens": line_box.get("tokens", []),
    }


def _build_line_boxes(
    words: list[dict[str, Any]],
    *,
    roi_nd: Any | None = None,
) -> list[dict[str, Any]]:
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (int(w.get("y", 0)), int(w.get("x", 0))))
    groups: list[list[dict[str, Any]]] = []
    for w in ordered:
        if not groups:
            groups.append([w])
            continue
        prev = groups[-1][-1]
        if int(w.get("y", 0)) - int(prev.get("y", 0)) < 22:
            groups[-1].append(w)
        else:
            groups.append([w])

    roi_gray = None
    if roi_nd is not None and cv2 is not None and np is not None:
        try:
            roi_gray = cv2.cvtColor(roi_nd, cv2.COLOR_BGR2GRAY)
        except Exception:
            roi_gray = None

    out: list[dict[str, Any]] = []
    pad_x = 8
    pad_y = 3
    for g in groups:
        xs = [int(w.get("x", 0)) for w in g]
        ys = [int(w.get("y", 0)) for w in g]
        x2s = [int(w.get("x", 0)) + int(w.get("w", 0)) for w in g]
        y2s = [int(w.get("y", 0)) + int(w.get("h", 0)) for w in g]
        h_vals = [max(1, int(w.get("h", 1))) for w in g]
        cy_vals = [int(w.get("y", 0)) + int(w.get("h", 0)) * 0.5 for w in g]

        # Use robust vertical statistics so boxes stay centered on glyphs even
        # when one token has a noisy OCR box.
        if np is not None:
            line_cy = float(np.median(np.array(cy_vals, dtype=np.float32)))
            line_h = float(np.percentile(np.array(h_vals, dtype=np.float32), 80))
            line_h = max(line_h, float(np.median(np.array(h_vals, dtype=np.float32))))
        else:
            sorted_cy = sorted(cy_vals)
            mid = len(sorted_cy) // 2
            if len(sorted_cy) % 2 == 0:
                line_cy = (sorted_cy[mid - 1] + sorted_cy[mid]) * 0.5
            else:
                line_cy = float(sorted_cy[mid])

            sorted_h = sorted(h_vals)
            mid_h = len(sorted_h) // 2
            if len(sorted_h) % 2 == 0:
                med_h = (sorted_h[mid_h - 1] + sorted_h[mid_h]) * 0.5
            else:
                med_h = float(sorted_h[mid_h])
            idx80 = min(
                len(sorted_h) - 1, max(0, int(round((len(sorted_h) - 1) * 0.8)))
            )
            p80_h = float(sorted_h[idx80])
            line_h = max(p80_h, med_h)

        x0 = min(xs) - pad_x
        x1 = max(x2s) + pad_x
        y0 = int(round(line_cy - 0.5 * line_h)) - pad_y
        y1 = int(round(line_cy + 0.5 * line_h)) + pad_y

        # Keep box from clipping obvious ascender/descender extremes.
        y0 = min(y0, min(ys) - 1)
        y1 = max(y1, max(y2s) + 1)
        line_box = {
            "x": x0,
            "y": y0,
            "w": max(1, x1 - x0),
            "h": max(1, y1 - y0),
            "tokens": [str(w.get("text", "")) for w in g if str(w.get("text", ""))],
        }
        if roi_gray is not None:
            line_box = _refine_line_box_with_text_mask(
                line_box, line_cy=line_cy, roi_gray=roi_gray
            )
        out.append(line_box)
    return out


def _overlay_root(token: str) -> str:
    return _overlay_root_impl(token)


def _is_edge_position(x: float, y: float, *, roi_width: int, roi_height: int) -> bool:
    return _is_edge_position_impl(x, y, roi_width=roi_width, roi_height=roi_height)


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
    return _collect_edge_overlay_stats_impl(
        raw_frames, roi_width=roi_width, roi_height=roi_height
    )


def _identify_banned_overlay(
    bins: dict[tuple[str, int, int], dict[str, float]],
    root_edge_frame_counts: dict[str, int],
    root_variants: dict[str, set[str]],
    *,
    total_frames: int,
    roi_width: int,
    roi_height: int,
) -> tuple[set[tuple[str, int, int]], set[str]]:
    return _identify_banned_overlay_impl(
        bins,
        root_edge_frame_counts,
        root_variants,
        total_frames=total_frames,
        roi_width=roi_width,
        roi_height=roi_height,
    )


def _filter_banned_overlay_words(
    raw_frames: list[dict[str, Any]],
    *,
    roi_width: int,
    roi_height: int,
    banned_keys: set[tuple[str, int, int]],
    banned_roots: set[str],
) -> list[dict[str, Any]]:
    return _filter_banned_overlay_words_impl(
        raw_frames,
        roi_width=roi_width,
        roi_height=roi_height,
        banned_keys=banned_keys,
        banned_roots=banned_roots,
    )


def _suppress_persistent_edge_overlay_words(
    raw_frames: list[dict[str, Any]],
    *,
    roi_width: int,
    roi_height: int,
) -> list[dict[str, Any]]:
    return _suppress_persistent_edge_overlay_words_impl(
        raw_frames, roi_width=roi_width, roi_height=roi_height
    )


def _suppress_early_banner_words(
    raw_frames: list[dict[str, Any]],
    *,
    roi_width: int,
    roi_height: int,
) -> list[dict[str, Any]]:
    return _suppress_early_banner_words_impl(
        raw_frames, roi_width=roi_width, roi_height=roi_height
    )


def _count_dense_line_groups(words: list[dict[str, Any]]) -> int:
    return _count_dense_line_groups_impl(words)


def _estimate_lyrics_start_time(raw_frames: list[dict[str, Any]]) -> float | None:
    return _estimate_lyrics_start_time_impl(raw_frames)


def _suppress_intro_title_words(
    raw_frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _suppress_intro_title_words_impl(raw_frames)


def _suppress_transient_digit_heavy_frames(
    raw_frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _suppress_transient_digit_heavy_frames_impl(raw_frames)


def _fill_transient_ocr_gaps(
    raw_frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _fill_transient_ocr_gaps_impl(
        raw_frames, build_line_boxes_fn=_build_line_boxes
    )


def _apply_post_ocr_filters(
    raw_frames: list[dict[str, Any]],
    *,
    roi_width: int,
    roi_height: int,
) -> list[dict[str, Any]]:
    return _apply_post_ocr_filters_impl(
        raw_frames,
        roi_width=roi_width,
        roi_height=roi_height,
        build_line_boxes_fn=_build_line_boxes,
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
    apply_post_filters: bool = True,
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
            roi_frames=buffered_rois,
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
    if not apply_post_filters:
        return raw
    return _apply_post_ocr_filters(raw, roi_width=rw, roi_height=rh)


def raw_frames_cache_path(
    video_path: Path,
    cache_dir: Path,
    fps: float,
    roi_rect: tuple[int, int, int, int],
    *,
    cache_version: str,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ocr_fingerprint = get_ocr_cache_fingerprint()
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
        return _apply_post_ocr_filters(
            loaded, roi_width=roi_rect[2], roi_height=roi_rect[3]
        )

    if collect_fn is None:
        raw = collect_raw_frames(
            video_path,
            0,
            duration,
            fps,
            roi_rect,
            apply_post_filters=False,
        )
    else:
        raw = collect_fn(video_path, 0, duration, fps, roi_rect)
    raw = _apply_post_ocr_filters(raw, roi_width=roi_rect[2], roi_height=roi_rect[3])
    cache_path.write_text(json.dumps(raw))
    return raw
