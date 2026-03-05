"""Post-OCR filtering and transient-gap interpolation helpers."""

from __future__ import annotations

import re
from typing import Any, Callable


def overlay_root(token: str) -> str:
    compact = "".join(ch for ch in token.lower() if ch.isalnum())
    return compact[:4]


def is_edge_position(x: float, y: float, *, roi_width: int, roi_height: int) -> bool:
    return x >= roi_width * 0.75 or y >= roi_height * 0.86


def collect_edge_overlay_stats(
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
            root = overlay_root(compact)
            if not root:
                continue
            x = float(w.get("x", 0.0))
            y = float(w.get("y", 0.0))
            if not is_edge_position(x, y, roi_width=roi_width, roi_height=roi_height):
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


def identify_banned_overlay(
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
            and is_edge_position(
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


def filter_banned_overlay_words(
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
            root = overlay_root(compact)
            x = float(w.get("x", 0.0))
            y = float(w.get("y", 0.0))
            key = (root, int(round(x / 24.0)), int(round(y / 24.0)))
            is_edge = is_edge_position(x, y, roi_width=roi_width, roi_height=roi_height)
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


def suppress_persistent_edge_overlay_words(
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
    bins, root_edge_frame_counts, root_variants = collect_edge_overlay_stats(
        raw_frames, roi_width=roi_width, roi_height=roi_height
    )
    banned_keys, banned_roots = identify_banned_overlay(
        bins,
        root_edge_frame_counts,
        root_variants,
        total_frames=total_frames,
        roi_width=roi_width,
        roi_height=roi_height,
    )
    if not banned_keys and not banned_roots:
        return raw_frames

    return filter_banned_overlay_words(
        raw_frames,
        roi_width=roi_width,
        roi_height=roi_height,
        banned_keys=banned_keys,
        banned_roots=banned_roots,
    )


def suppress_early_banner_words(
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


def count_dense_line_groups(words: list[dict[str, Any]]) -> int:
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


def estimate_lyrics_start_time(raw_frames: list[dict[str, Any]]) -> float | None:
    hits: list[float] = []
    for fr in raw_frames:
        words = [w for w in fr.get("words", []) if isinstance(w, dict)]
        if len(words) < 5:
            continue
        if count_dense_line_groups(words) < 2:
            continue
        hits.append(float(fr.get("time", 0.0)))
    if len(hits) < 3:
        return None
    plausible = [t for t in hits if 8.0 <= t <= 45.0]
    if len(plausible) < 3:
        return None
    return min(plausible)


def suppress_intro_title_words(
    raw_frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(raw_frames) < 30:
        return raw_frames
    lyrics_start = estimate_lyrics_start_time(raw_frames)
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


def suppress_transient_digit_heavy_frames(
    raw_frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(raw_frames) < 3:
        return raw_frames

    out = [dict(fr) for fr in raw_frames]
    for i, fr in enumerate(raw_frames):
        if _is_transient_digit_heavy_frame(raw_frames, i):
            out[i] = {**fr, "words": [], "line_boxes": []}
    return out


def _frame_words(frame: dict[str, Any]) -> list[dict[str, Any]]:
    return [w for w in frame.get("words", []) if isinstance(w, dict)]


def _token_texts(words: list[dict[str, Any]]) -> list[str]:
    return [text for text in (str(w.get("text", "")).strip() for w in words) if text]


def _digit_heavy_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    digit_heavy = sum(1 for token in tokens if any(ch.isdigit() for ch in token))
    return digit_heavy / float(len(tokens))


def _normalized_token_set_from_texts(tokens: list[str]) -> set[str]:
    return {
        tok for tok in (re.sub(r"[^a-z0-9]+", "", t.lower()) for t in tokens) if tok
    }


def _normalized_token_set_from_words(words: list[dict[str, Any]]) -> set[str]:
    return {
        tok
        for tok in (
            re.sub(r"[^a-z0-9]+", "", str(word.get("text", "")).lower())
            for word in words
        )
        if tok
    }


def _is_dense_neighbor_triplet(
    prev: dict[str, Any], current: dict[str, Any], nxt: dict[str, Any]
) -> bool:
    t_cur = float(current.get("time", 0.0))
    t_prev = float(prev.get("time", t_cur))
    t_next = float(nxt.get("time", t_cur))
    return (t_cur - t_prev) <= 0.6 and (t_next - t_cur) <= 0.6


def _is_transient_digit_heavy_frame(raw_frames: list[dict[str, Any]], idx: int) -> bool:
    if idx <= 0 or idx >= (len(raw_frames) - 1):
        return False
    current = raw_frames[idx]
    current_words = _frame_words(current)
    if len(current_words) < 3:
        return False
    current_tokens = _token_texts(current_words)
    if len(current_tokens) < 3 or _digit_heavy_ratio(current_tokens) < 0.4:
        return False

    prev = raw_frames[idx - 1]
    nxt = raw_frames[idx + 1]
    prev_words = _frame_words(prev)
    next_words = _frame_words(nxt)
    if len(prev_words) < 6 or len(next_words) < 6:
        return False
    if not _is_dense_neighbor_triplet(prev, current, nxt):
        return False

    current_norm = _normalized_token_set_from_texts(current_tokens)
    if not current_norm:
        return False
    neighbor_norm = _normalized_token_set_from_words(
        prev_words
    ) | _normalized_token_set_from_words(next_words)
    overlap = len(current_norm & neighbor_norm) / float(len(current_norm))
    return overlap <= 0.2


def _find_word_match(
    target_w: dict[str, Any], candidates: list[Any]
) -> dict[str, Any] | None:
    tx = float(target_w.get("x", 0))
    ty = float(target_w.get("y", 0))
    ttext = str(target_w.get("text", ""))
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        cx = float(cand.get("x", 0))
        cy = float(cand.get("y", 0))
        if abs(cx - tx) > 20 or abs(cy - ty) > 10:
            continue
        if str(cand.get("text", "")) == ttext:
            return cand
    return None


def _is_dense_triplet(t_prev: float, t_curr: float, t_next: float) -> bool:
    return (t_curr - t_prev) <= 0.6 and (t_next - t_curr) <= 0.6


def _frame_word_list(frame: dict[str, Any]) -> list[Any] | None:
    words = frame.get("words", [])
    return words if isinstance(words, list) else None


def _interpolate_missing_word(pw: dict[str, Any], nw: dict[str, Any]) -> dict[str, Any]:
    px, py = float(pw.get("x", 0)), float(pw.get("y", 0))
    nx, ny = float(nw.get("x", 0)), float(nw.get("y", 0))
    pw_w, pw_h = float(pw.get("w", 0)), float(pw.get("h", 0))
    nw_w, nw_h = float(nw.get("w", 0)), float(nw.get("h", 0))
    pb = float(pw.get("brightness", 0))
    nb = float(nw.get("brightness", 0))
    pd = float(pw.get("density", 0))
    nd = float(nw.get("density", 0))
    return {
        "text": nw.get("text"),
        "x": int((px + nx) * 0.5),
        "y": int((py + ny) * 0.5),
        "w": int((pw_w + nw_w) * 0.5),
        "h": int((pw_h + nw_h) * 0.5),
        "brightness": (pb + nb) * 0.5,
        "density": (pd + nd) * 0.5,
    }


def fill_transient_ocr_gaps(
    raw_frames: list[dict[str, Any]],
    *,
    build_line_boxes_fn: Callable[..., list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if len(raw_frames) < 3:
        return raw_frames

    out = [dict(fr) for fr in raw_frames]

    for i in range(1, len(raw_frames) - 1):
        prev_f = raw_frames[i - 1]
        next_f = raw_frames[i + 1]
        curr_f = out[i]

        t_prev = float(prev_f.get("time", 0.0))
        t_curr = float(curr_f.get("time", 0.0))
        t_next = float(next_f.get("time", 0.0))

        if not _is_dense_triplet(t_prev, t_curr, t_next):
            continue

        prev_words = _frame_word_list(prev_f)
        next_words = _frame_word_list(next_f)
        curr_words = _frame_word_list(curr_f)
        if prev_words is None or next_words is None or curr_words is None:
            continue

        injected_words = []
        for pw in prev_words:
            if not isinstance(pw, dict):
                continue

            nw = _find_word_match(pw, next_words)
            if not nw:
                continue

            cw = _find_word_match(pw, curr_words)
            if cw:
                continue
            injected_words.append(_interpolate_missing_word(pw, nw))

        if injected_words:
            combined = list(curr_words) + injected_words
            out[i] = {
                **curr_f,
                "words": combined,
                "line_boxes": build_line_boxes_fn(combined, roi_nd=None),
            }

    return out


def apply_post_ocr_filters(
    raw_frames: list[dict[str, Any]],
    *,
    roi_width: int,
    roi_height: int,
    build_line_boxes_fn: Callable[..., list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    filtered = suppress_persistent_edge_overlay_words(
        raw_frames, roi_width=roi_width, roi_height=roi_height
    )
    filtered = suppress_early_banner_words(
        filtered, roi_width=roi_width, roi_height=roi_height
    )
    filtered = suppress_intro_title_words(filtered)
    filtered = suppress_transient_digit_heavy_frames(filtered)
    return fill_transient_ocr_gaps(filtered, build_line_boxes_fn=build_line_boxes_fn)
