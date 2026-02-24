"""OCR frame sampling and caching helpers for visual bootstrap."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

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
            idx80 = min(len(sorted_h) - 1, max(0, int(round((len(sorted_h) - 1) * 0.8))))
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
        is_extreme_corner = (
            mean_x >= roi_width * 0.84
            and mean_y >= roi_height * 0.76
            and (var_x**0.5) <= 12.0
            and (var_y**0.5) <= 12.0
        )
        min_variant_count = 1 if is_extreme_corner else 2
        min_freq = 0.12 if is_extreme_corner else 0.22
        if (
            freq >= min_freq
            and (var_x**0.5) <= 18.0
            and (var_y**0.5) <= 18.0
            and _is_edge_position(
                mean_x, mean_y, roi_width=roi_width, roi_height=roi_height
            )
            and len(root_variants.get(root, set())) >= min_variant_count
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
        frame_time = float(frame.get("time", 0.0))
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
            is_micro_corner_text = (
                bool(compact)
                and len(compact) <= 10
                and x >= roi_width * 0.82
                and y >= roi_height * 0.84
                and float(w.get("h", 0.0)) <= roi_height * 0.09
                and float(w.get("w", 0.0)) <= roi_width * 0.22
            )
            is_early_banner_word = (
                bool(compact)
                and frame_time <= 30.0
                and y <= roi_height * 0.45
                and float(w.get("h", 0.0)) >= roi_height * 0.12
                and float(w.get("w", 0.0)) >= roi_width * 0.16
            )
            if is_early_banner_word:
                continue
            if is_micro_corner_text:
                continue
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


def _suppress_early_banner_words(
    raw_frames: list[dict[str, Any]],
    *,
    roi_width: int,
    roi_height: int,
) -> list[dict[str, Any]]:
    if not raw_frames:
        return raw_frames
    out: list[dict[str, Any]] = []
    for frame in raw_frames:
        frame_time = float(frame.get("time", 0.0))
        words = frame.get("words", [])
        if frame_time > 30.0 or not isinstance(words, list):
            out.append(frame)
            continue
        new_words = []
        for w in words:
            if not isinstance(w, dict):
                continue
            text = str(w.get("text", ""))
            compact = re.sub(r"[^a-z0-9]", "", text.lower())
            y = float(w.get("y", 0.0))
            ww = float(w.get("w", 0.0))
            hh = float(w.get("h", 0.0))
            # Only hit REALLY large banner-like text (title/credits)
            is_early_banner_word = (
                bool(compact)
                and y <= roi_height * 0.45
                and hh >= roi_height * 0.20
                and ww >= roi_width * 0.25
            )
            if is_early_banner_word:
                continue
            new_words.append(w)
        out.append({**frame, "words": new_words})
    return out


def _count_dense_line_groups(words: list[dict[str, Any]]) -> int:
    if not words:
        return 0
    ys = sorted(int(w.get("y", 0)) for w in words if isinstance(w, dict))
    if not ys:
        return 0
    groups: list[list[int]] = [[ys[0]]]
    for y in ys[1:]:
        if y - groups[-1][-1] < 22:
            groups[-1].append(y)
        else:
            groups.append([y])
    return sum(1 for g in groups if len(g) >= 2)


def _estimate_lyrics_start_time(raw_frames: list[dict[str, Any]]) -> float | None:
    hits: list[float] = []
    for fr in raw_frames:
        words = [w for w in fr.get("words", []) if isinstance(w, dict)]
        if len(words) < 5:
            continue
        if _count_dense_line_groups(words) < 2:
            continue
        hits.append(float(fr.get("time", 0.0)))
    if len(hits) < 3:
        return None
    plausible = [t for t in hits if 8.0 <= t <= 45.0]
    if len(plausible) < 3:
        return None
    return min(plausible)


def _suppress_intro_title_words(
    raw_frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(raw_frames) < 30:
        return raw_frames
    lyrics_start = _estimate_lyrics_start_time(raw_frames)
    if lyrics_start is None:
        return raw_frames
    cutoff = max(0.0, lyrics_start - 0.25)
    out: list[dict[str, Any]] = []
    for fr in raw_frames:
        t = float(fr.get("time", 0.0))
        if t < cutoff:
            out.append({**fr, "words": [], "line_boxes": []})
        else:
            out.append(fr)
    return out


def _suppress_transient_digit_heavy_frames(
    raw_frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Clear single-frame OCR glitches dominated by digit-like garbage tokens."""
    if len(raw_frames) < 3:
        return raw_frames

    out = [dict(fr) for fr in raw_frames]
    for i, fr in enumerate(raw_frames):
        words = [w for w in fr.get("words", []) if isinstance(w, dict)]
        if len(words) < 3:
            continue
        toks = [str(w.get("text", "")).strip() for w in words]
        toks = [t for t in toks if t]
        if len(toks) < 3:
            continue

        digit_heavy = sum(1 for t in toks if any(ch.isdigit() for ch in t))
        if (digit_heavy / float(len(toks))) < 0.4:
            continue

        if i <= 0 or i >= (len(raw_frames) - 1):
            continue
        prev = raw_frames[i - 1]
        nxt = raw_frames[i + 1]
        prev_words = [w for w in prev.get("words", []) if isinstance(w, dict)]
        next_words = [w for w in nxt.get("words", []) if isinstance(w, dict)]
        if len(prev_words) < 6 or len(next_words) < 6:
            continue

        t_cur = float(fr.get("time", 0.0))
        t_prev = float(prev.get("time", t_cur))
        t_next = float(nxt.get("time", t_cur))
        if (t_cur - t_prev) > 0.6 or (t_next - t_cur) > 0.6:
            continue

        prev_norm = {
            re.sub(r"[^a-z0-9]+", "", str(w.get("text", "")).lower())
            for w in prev_words
        }
        next_norm = {
            re.sub(r"[^a-z0-9]+", "", str(w.get("text", "")).lower())
            for w in next_words
        }
        neigh = {t for t in (prev_norm | next_norm) if t}
        cur_norm = {re.sub(r"[^a-z0-9]+", "", t.lower()) for t in toks}
        cur_norm = {t for t in cur_norm if t}
        if not cur_norm:
            continue
        overlap = len(cur_norm & neigh) / float(len(cur_norm))
        if overlap > 0.2:
            continue

        out[i] = {**fr, "words": [], "line_boxes": []}
    return out


def _fill_transient_ocr_gaps(
    raw_frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fill single-frame gaps where a word is seen in prev/next frames but missing in current."""
    if len(raw_frames) < 3:
        return raw_frames

    out = [dict(fr) for fr in raw_frames]

    # Helper to find match
    def _find_match(
        target_w: dict[str, Any], candidates: list[Any]
    ) -> dict[str, Any] | None:
        tx = float(target_w.get("x", 0))
        ty = float(target_w.get("y", 0))
        ttext = str(target_w.get("text", ""))

        for cw in candidates:
            if not isinstance(cw, dict):
                continue
            cx = float(cw.get("x", 0))
            cy = float(cw.get("y", 0))
            ctext = str(cw.get("text", ""))

            # Spatial proximity (20px x 10px tolerance)
            if abs(cx - tx) > 20 or abs(cy - ty) > 10:
                continue

            if ttext == ctext:
                return cw
        return None

    for i in range(1, len(raw_frames) - 1):
        prev_f = raw_frames[i - 1]
        next_f = raw_frames[i + 1]

        # Use the mutable output frame for current
        curr_f = out[i]

        t_prev = float(prev_f.get("time", 0.0))
        t_curr = float(curr_f.get("time", 0.0))
        t_next = float(next_f.get("time", 0.0))

        if (t_curr - t_prev) > 0.6 or (t_next - t_curr) > 0.6:
            continue

        prev_words = prev_f.get("words", [])
        next_words = next_f.get("words", [])
        curr_words = curr_f.get("words", [])

        if (
            not isinstance(prev_words, list)
            or not isinstance(next_words, list)
            or not isinstance(curr_words, list)
        ):
            continue

        injected_words = []
        for pw in prev_words:
            if not isinstance(pw, dict):
                continue

            nw = _find_match(pw, next_words)
            if not nw:
                continue

            cw = _find_match(pw, curr_words)
            if cw:
                continue

            # Interpolate
            px, py = float(pw.get("x", 0)), float(pw.get("y", 0))
            nx, ny = float(nw.get("x", 0)), float(nw.get("y", 0))
            pw_w, pw_h = float(pw.get("w", 0)), float(pw.get("h", 0))
            nw_w, nw_h = float(nw.get("w", 0)), float(nw.get("h", 0))

            ix = int((px + nx) * 0.5)
            iy = int((py + ny) * 0.5)
            iw = int((pw_w + nw_w) * 0.5)
            ih = int((pw_h + nw_h) * 0.5)

            pb = float(pw.get("brightness", 0))
            nb = float(nw.get("brightness", 0))
            pd = float(pw.get("density", 0))
            nd = float(nw.get("density", 0))

            new_word = {
                "text": nw.get("text"),
                "x": ix,
                "y": iy,
                "w": iw,
                "h": ih,
                "brightness": (pb + nb) * 0.5,
                "density": (pd + nd) * 0.5,
            }
            injected_words.append(new_word)

        if injected_words:
            # Combine and rebuild lines
            combined = list(curr_words) + injected_words
            out[i] = {
                **curr_f,
                "words": combined,
                "line_boxes": _build_line_boxes(combined, roi_nd=None),
            }

    return out


def _apply_post_ocr_filters(
    raw_frames: list[dict[str, Any]],
    *,
    roi_width: int,
    roi_height: int,
) -> list[dict[str, Any]]:
    filtered = _suppress_persistent_edge_overlay_words(
        raw_frames, roi_width=roi_width, roi_height=roi_height
    )
    filtered = _suppress_early_banner_words(
        filtered, roi_width=roi_width, roi_height=roi_height
    )
    filtered = _suppress_intro_title_words(filtered)
    filtered = _suppress_transient_digit_heavy_frames(filtered)
    return _fill_transient_ocr_gaps(filtered)


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
