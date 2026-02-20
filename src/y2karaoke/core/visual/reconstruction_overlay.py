"""Static overlay suppression helpers for visual lyric reconstruction."""

from __future__ import annotations

from typing import Any

from ..text_utils import normalize_text_basic

_OVERLAY_BIN_PX = 24.0
_OVERLAY_MAX_JITTER_PX = 20.0


def _filter_static_overlay_words(
    raw_frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not raw_frames:
        return raw_frames

    total_frames = len(raw_frames)
    if total_frames < 20:
        return raw_frames
    all_y, stats, root_frame_counts, early_root_frame_counts, root_variant_counts = (
        _collect_overlay_stats(raw_frames)
    )

    if not all_y:
        return raw_frames
    y_min = min(all_y)
    y_max = max(all_y)
    if (y_max - y_min) < 60.0:
        return raw_frames

    all_x = [
        float(w.get("x", 0.0))
        for fr in raw_frames
        for w in fr.get("words", [])
        if isinstance(w, dict)
    ]
    x_max = max(all_x) if all_x else 0.0
    y_bottom_cut = y_min + 0.82 * (y_max - y_min)

    static_keys = _identify_static_overlay_keys(
        stats, total_frames, y_min=y_min, y_max=y_max
    )
    overlay_roots = _infer_overlay_roots(
        root_frame_counts,
        early_root_frame_counts,
        root_variant_counts,
        total_frames=total_frames,
    )

    if not static_keys and not overlay_roots:
        return raw_frames

    out: list[dict[str, Any]] = []
    for frame in raw_frames:
        new_words = []
        for w in frame.get("words", []):
            tok = normalize_text_basic(str(w.get("text", ""))).strip()
            tok_compact = "".join(ch for ch in tok.lower() if ch.isalnum())
            y_val = float(w.get("y", 0.0))
            x_val = float(w.get("x", 0.0))
            is_short_bottom_right = (
                len(tok_compact) <= 4
                and y_val >= y_bottom_cut
                and x_val >= (0.55 * x_max if x_max > 0 else x_val + 1)
            )
            if is_short_bottom_right:
                continue

            root = _overlay_token_root(tok)
            if root is None:
                new_words.append(w)
                continue
            key = (
                root,
                int(round(float(w.get("x", 0.0)) / _OVERLAY_BIN_PX)),
                int(round(float(w.get("y", 0.0)) / _OVERLAY_BIN_PX)),
            )
            if key in static_keys or root in overlay_roots or is_short_bottom_right:
                continue
            new_words.append(w)
        out.append({**frame, "words": new_words})
    return out


def _collect_overlay_stats(
    raw_frames: list[dict[str, Any]],
) -> tuple[
    list[float],
    dict[tuple[str, int, int], dict[str, float]],
    dict[str, int],
    dict[str, int],
    dict[str, int],
]:
    all_y: list[float] = []
    stats: dict[tuple[str, int, int], dict[str, float]] = {}
    root_frame_counts: dict[str, int] = {}
    early_root_frame_counts: dict[str, int] = {}
    root_variant_sets: dict[str, set[str]] = {}
    first_time = float(raw_frames[0].get("time", 0.0))
    early_limit = first_time + 35.0
    for frame in raw_frames:
        seen: set[tuple[str, int, int]] = set()
        seen_roots: set[str] = set()
        frame_time = float(frame.get("time", first_time))
        for w in frame.get("words", []):
            try:
                x = float(w["x"])
                y = float(w["y"])
            except Exception:
                continue
            all_y.append(y)
            tok = normalize_text_basic(str(w.get("text", ""))).strip()
            root = _overlay_token_root(tok)
            if root is None:
                continue
            compact = "".join(ch for ch in tok.lower() if ch.isalnum())
            if compact:
                root_variant_sets.setdefault(root, set()).add(compact)
            seen_roots.add(root)
            key = (
                root,
                int(round(x / _OVERLAY_BIN_PX)),
                int(round(y / _OVERLAY_BIN_PX)),
            )
            if key in seen:
                continue
            seen.add(key)
            rec = stats.setdefault(
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
        for root in seen_roots:
            root_frame_counts[root] = root_frame_counts.get(root, 0) + 1
            if frame_time <= early_limit:
                early_root_frame_counts[root] = early_root_frame_counts.get(root, 0) + 1
    root_variant_counts = {k: len(v) for k, v in root_variant_sets.items()}
    return all_y, stats, root_frame_counts, early_root_frame_counts, root_variant_counts


def _overlay_token_root(token: str) -> str | None:
    compact = "".join(ch for ch in token.lower() if ch.isalnum())
    if len(compact) < 4:
        return None
    return compact[:4]


def _identify_static_overlay_keys(
    stats: dict[tuple[str, int, int], dict[str, float]],
    total_frames: int,
    *,
    y_min: float,
    y_max: float,
) -> set[tuple[str, int, int]]:
    y_top_zone = y_min + 0.15 * (y_max - y_min)
    y_bottom_zone = y_min + 0.85 * (y_max - y_min)

    static_keys: set[tuple[str, int, int]] = set()
    for key, rec in stats.items():
        n = max(rec["count"], 1.0)
        freq = rec["count"] / max(float(total_frames), 1.0)
        mean_x = rec["sum_x"] / n
        mean_y = rec["sum_y"] / n
        var_x = max(rec["sum_x2"] / n - mean_x * mean_x, 0.0)
        var_y = max(rec["sum_y2"] / n - mean_y * mean_y, 0.0)
        std_x = var_x**0.5
        std_y = var_y**0.5

        is_stable = std_x <= 8.0 and std_y <= 8.0
        # Only consider extreme top/bottom zones to avoid lyric collisions
        is_in_overlay_zone = mean_y <= y_top_zone or mean_y >= y_bottom_zone

        if freq >= 0.7 and is_stable and is_in_overlay_zone:
            static_keys.add(key)
    return static_keys


def _infer_overlay_roots(
    root_frame_counts: dict[str, int],
    early_root_frame_counts: dict[str, int],
    root_variant_counts: dict[str, int],
    *,
    total_frames: int,
) -> set[str]:
    if total_frames <= 0:
        return set()
    out: set[str] = set()
    for root, count in root_frame_counts.items():
        total_cov = count / float(total_frames)
        early_cov = early_root_frame_counts.get(root, 0) / float(max(1, total_frames))
        variants = root_variant_counts.get(root, 0)
        if total_cov >= 0.25 and early_cov >= 0.15 and variants >= 3:
            out.add(root)
    return out
