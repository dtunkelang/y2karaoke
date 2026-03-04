"""OCR frame sampling and caching helpers for visual bootstrap."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .bootstrap_ocr_cache import (
    collect_raw_frames_cached as _collect_raw_frames_cached_impl,
)
from .bootstrap_ocr_cache import (
    raw_frames_cache_path as _raw_frames_cache_path_impl,
)
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
from .bootstrap_ocr_line_boxes import (
    build_line_boxes as _build_line_boxes_impl,
)
from .bootstrap_ocr_line_boxes import (
    get_word_ink_metrics as _get_word_ink_metrics_impl,
)
from .bootstrap_ocr_line_boxes import (
    refine_line_box_with_text_mask as _refine_line_box_with_text_mask_impl,
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
    return _get_word_ink_metrics_impl(roi_nd, x=x, y=y, w=w, h=h)


def _refine_line_box_with_text_mask(
    line_box: dict[str, Any],
    *,
    line_cy: float,
    roi_gray: Any,
) -> dict[str, Any]:
    return _refine_line_box_with_text_mask_impl(
        line_box,
        line_cy=line_cy,
        roi_gray=roi_gray,
    )


def _build_line_boxes(
    words: list[dict[str, Any]],
    *,
    roi_nd: Any | None = None,
) -> list[dict[str, Any]]:
    return _build_line_boxes_impl(words, roi_nd=roi_nd)


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
    return _raw_frames_cache_path_impl(
        video_path,
        cache_dir,
        fps,
        roi_rect,
        cache_version=cache_version,
        ocr_fingerprint_fn=get_ocr_cache_fingerprint,
    )


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
    return _collect_raw_frames_cached_impl(
        video_path,
        duration,
        fps,
        roi_rect,
        cache_dir,
        cache_version=cache_version,
        log_fn=log_fn,
        collect_fn=collect_fn,
        raw_frames_cache_path_fn=raw_frames_cache_path,
        collect_raw_frames_fn=collect_raw_frames,
        apply_post_ocr_filters_fn=_apply_post_ocr_filters,
    )
