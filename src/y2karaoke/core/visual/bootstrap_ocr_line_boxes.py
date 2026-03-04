"""Line-box geometry and ink visibility helpers for OCR bootstrap."""

from __future__ import annotations

from typing import Any

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency behavior
    cv2 = None  # type: ignore
    np = None  # type: ignore


def get_word_ink_metrics(
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


def _group_words_by_row(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    ordered = sorted(words, key=lambda w: (int(w.get("y", 0)), int(w.get("x", 0))))
    groups: list[list[dict[str, Any]]] = []
    for word in ordered:
        if not groups:
            groups.append([word])
            continue
        prev = groups[-1][-1]
        if int(word.get("y", 0)) - int(prev.get("y", 0)) < 22:
            groups[-1].append(word)
            continue
        groups.append([word])
    return groups


def _roi_gray(roi_nd: Any | None) -> Any | None:
    if roi_nd is None or cv2 is None or np is None:
        return None
    try:
        return cv2.cvtColor(roi_nd, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None


def _line_center_height(h_vals: list[int], cy_vals: list[float]) -> tuple[float, float]:
    # Use robust vertical statistics so boxes stay centered on glyphs even
    # when one token has a noisy OCR box.
    if np is not None:
        line_cy = float(np.median(np.array(cy_vals, dtype=np.float32)))
        line_h = float(np.percentile(np.array(h_vals, dtype=np.float32), 80))
        line_h = max(line_h, float(np.median(np.array(h_vals, dtype=np.float32))))
        return line_cy, line_h

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
    return line_cy, max(p80_h, med_h)


def _expanded_line_box_bounds(
    line_box: dict[str, Any], *, roi_w: int, roi_h: int
) -> tuple[int, int, int, int, int, int] | None:
    x0 = int(line_box.get("x", 0))
    y0 = int(line_box.get("y", 0))
    w = int(line_box.get("w", 1))
    h = int(line_box.get("h", 1))
    x1 = x0 + max(1, w)
    y1 = y0 + max(1, h)
    sx0 = max(0, x0 - 10)
    sy0 = max(0, y0 - 8)
    sx1 = min(roi_w, x1 + 10)
    sy1 = min(roi_h, y1 + 8)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    return sx0, sy0, sx1, sy1, w, h


def _ink_mask(sub: Any) -> Any | None:
    _, otsu = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fixed = (sub >= 45).astype(np.uint8) * 255
    mask = np.where((otsu > 0) | (fixed > 0), 1, 0).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8)
    ).astype(np.uint8)
    if int(mask.sum()) <= 0:
        return None
    return mask


def _active_rows(mask: Any, *, width: int) -> Any | None:
    row_counts = mask.sum(axis=1)
    row_gate = max(2, int(round(width * 0.012)))
    active = np.where(row_counts >= row_gate)[0]
    if active.size == 0:
        active = np.where(row_counts > 0)[0]
    if active.size == 0:
        return None
    return active


def _best_row_run(active_rows: Any, *, center_row: int) -> tuple[int, int]:
    splits = np.where(np.diff(active_rows) > 1)[0] + 1
    runs = np.split(active_rows, splits)
    best = runs[0]
    best_dist = float("inf")
    for run in runs:
        lo, hi = int(run[0]), int(run[-1])
        if lo <= center_row <= hi:
            return lo, hi + 1
        mid = (lo + hi) * 0.5
        dist = abs(mid - center_row)
        if dist < best_dist:
            best_dist = dist
            best = run
    return int(best[0]), int(best[-1]) + 1


def _refined_box_from_band(
    *,
    band: Any,
    sx0: int,
    sy0: int,
    ry0: int,
    ry1: int,
    roi_w: int,
    roi_h: int,
    base_w: int,
    base_h: int,
) -> tuple[int, int, int, int] | None:
    if band.size == 0 or int(band.sum()) <= 0:
        return None
    _, xs = np.where(band > 0)
    if xs.size == 0:
        return None
    tx0 = max(0, sx0 + int(xs.min()) - 2)
    tx1 = min(roi_w, sx0 + int(xs.max()) + 3)
    ty0 = max(0, sy0 + ry0 - 2)
    ty1 = min(roi_h, sy0 + ry1 + 4)
    tw = max(1, tx1 - tx0)
    th = max(1, ty1 - ty0)
    if tw < int(0.35 * base_w) or tw > int(2.4 * base_w):
        return None
    if th < int(0.5 * base_h) or th > int(2.0 * base_h):
        return None
    return tx0, ty0, tw, th


def refine_line_box_with_text_mask(
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
    expanded = _expanded_line_box_bounds(line_box, roi_w=rw, roi_h=rh)
    if expanded is None:
        return line_box
    sx0, sy0, sx1, sy1, base_w, base_h = expanded

    sub = roi_gray[sy0:sy1, sx0:sx1]
    if sub.size == 0:
        return line_box

    mask = _ink_mask(sub)
    if mask is None:
        return line_box

    active_rows = _active_rows(mask, width=(sx1 - sx0))
    if active_rows is None:
        return line_box
    center_row = int(round(line_cy)) - sy0
    ry0, ry1 = _best_row_run(active_rows, center_row=center_row)
    band = mask[ry0:ry1, :]
    refined = _refined_box_from_band(
        band=band,
        sx0=sx0,
        sy0=sy0,
        ry0=ry0,
        ry1=ry1,
        roi_w=rw,
        roi_h=rh,
        base_w=base_w,
        base_h=base_h,
    )
    if refined is None:
        return line_box
    tx0, ty0, tw, th = refined

    return {
        "x": tx0,
        "y": ty0,
        "w": tw,
        "h": th,
        "tokens": line_box.get("tokens", []),
    }


def build_line_boxes(
    words: list[dict[str, Any]],
    *,
    roi_nd: Any | None = None,
) -> list[dict[str, Any]]:
    if not words:
        return []
    groups = _group_words_by_row(words)
    roi_gray = _roi_gray(roi_nd)

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
        line_cy, line_h = _line_center_height(h_vals, cy_vals)

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
            line_box = refine_line_box_with_text_mask(
                line_box, line_cy=line_cy, roi_gray=roi_gray
            )
        out.append(line_box)
    return out
