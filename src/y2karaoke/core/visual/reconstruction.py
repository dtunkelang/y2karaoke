from __future__ import annotations

from typing import Any, Dict

from ..models import TargetLine
from ..text_utils import (
    normalize_ocr_line,
    normalize_ocr_tokens,
    normalize_text_basic,
    text_similarity,
)

_OVERLAY_BIN_PX = 24.0
_OVERLAY_MAX_JITTER_PX = 20.0
_LANE_PROXIMITY_PX = 18.0


def snap(value: float) -> float:
    # Assuming 0.05s snap from original tool
    return round(round(float(value) / 0.05) * 0.05, 3)


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
    y_top_cut = y_min + 0.35 * (y_max - y_min)
    y_bottom_cut = y_min + 0.82 * (y_max - y_min)
    all_x = [
        float(w.get("x", 0.0))
        for fr in raw_frames
        for w in fr.get("words", [])
        if isinstance(w, dict)
    ]
    x_max = max(all_x) if all_x else 0.0

    static_keys = _identify_static_overlay_keys(stats, total_frames, y_top_cut)
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
    y_top_cut: float,
) -> set[tuple[str, int, int]]:
    static_keys: set[tuple[str, int, int]] = set()
    for key, rec in stats.items():
        n = max(rec["count"], 1.0)
        freq = rec["count"] / max(float(total_frames), 1.0)
        mean_x = rec["sum_x"] / n
        mean_y = rec["sum_y"] / n
        var_x = max(rec["sum_x2"] / n - mean_x * mean_x, 0.0)
        var_y = max(rec["sum_y2"] / n - mean_y * mean_y, 0.0)
        if (
            freq >= 0.45
            and (var_x**0.5) <= _OVERLAY_MAX_JITTER_PX
            and (var_y**0.5) <= _OVERLAY_MAX_JITTER_PX
            and mean_y <= y_top_cut
        ):
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
        if total_cov >= 0.2 and early_cov >= 0.12 and variants >= 3:
            out.add(root)
    return out


def _suppress_short_duplicate_reentries(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(entries) < 2:
        return entries

    out: list[dict[str, Any]] = []
    for ent in entries:
        duration = float(ent["last"]) - float(ent["first"])
        if duration > 1.2:
            out.append(ent)
            continue

        is_dup_reentry = False
        for prev in reversed(out[-12:]):
            time_gap = float(ent["first"]) - float(prev["first"])
            if time_gap < 0:
                continue
            if time_gap > 20.0:
                break
            if not _is_same_lane(ent, prev):
                continue
            if text_similarity(ent["text"], prev["text"]) < 0.9:
                continue
            prev_duration = float(prev["last"]) - float(prev["first"])
            if prev_duration >= 2.0 or prev_duration >= duration + 0.8:
                is_dup_reentry = True
                break

        if not is_dup_reentry:
            out.append(ent)
    return out


def _merge_short_same_lane_reentries(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge short same-lane reentries caused by transient OCR drops."""
    if len(entries) < 2:
        return entries

    out: list[dict[str, Any]] = []

    for ent in entries:
        tokens = [t for t in ent.get("words", []) if str(t).strip()]
        duration = float(ent["last"]) - float(ent["first"])
        prev_idx: int | None = None
        for idx in range(len(out) - 1, max(-1, len(out) - 13), -1):
            prev = out[idx]
            if text_similarity(prev["text"], ent["text"]) < 0.9:
                continue
            if not _is_same_lane(prev, ent):
                continue
            prev_idx = idx
            break

        if prev_idx is not None and len(tokens) <= 2:
            prev = out[prev_idx]
            prev_tokens = [t for t in prev.get("words", []) if str(t).strip()]
            if len(prev_tokens) <= 2:
                gap = float(ent["first"]) - float(prev["last"])
                if 0.0 <= gap <= 4.0:
                    mids = out[prev_idx + 1 :]
                    has_lane_conflict = any(
                        _is_same_lane(mid, ent)
                        and text_similarity(mid["text"], ent["text"]) < 0.9
                        for mid in mids
                    )
                    prev_duration = float(prev["last"]) - float(prev["first"])
                    cross_lane_same_text = any(
                        not _is_same_lane(mid, ent)
                        and text_similarity(mid["text"], ent["text"]) >= 0.9
                        for mid in mids
                    )
                    allow_merge = prev_duration >= 1.0 or cross_lane_same_text
                    continuation_split = gap <= 1.5
                    if continuation_split and not has_lane_conflict:
                        prev["last"] = max(float(prev["last"]), float(ent["last"]))
                        if len(ent.get("w_rois", [])) > len(
                            prev.get("w_rois", [])
                        ) and ent.get("w_rois"):
                            prev["w_rois"] = ent["w_rois"]
                        continue
                    if duration <= 1.6 and allow_merge and not has_lane_conflict:
                        prev["last"] = max(float(prev["last"]), float(ent["last"]))
                        if len(ent.get("w_rois", [])) > len(
                            prev.get("w_rois", [])
                        ) and ent.get("w_rois"):
                            prev["w_rois"] = ent["w_rois"]
                        continue

        out.append(ent)

    return out


def _is_same_lane(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return abs(float(a.get("y", 0.0)) - float(b.get("y", 0.0))) <= _LANE_PROXIMITY_PX


_INTRO_META_KEYWORDS = {
    "karaoke",
    "singking",
    "version",
    "official",
    "records",
    "universal",
    "lyrics",
    "instrumental",
}

_LYRIC_FUNCTION_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "dont",
    "for",
    "i",
    "im",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "oh",
    "on",
    "so",
    "the",
    "to",
    "we",
    "yeah",
    "you",
    "your",
}


def _is_intro_artifact(entry: dict[str, Any]) -> bool:
    text = str(entry.get("text", ""))
    words = [w for w in entry.get("words", []) if str(w).strip()]
    duration = float(entry.get("last", 0.0)) - float(entry.get("first", 0.0))
    cleaned = normalize_text_basic(text).strip()
    compact_words = ["".join(ch for ch in w if ch.isalnum()) for w in cleaned.split()]
    compact_words = [w for w in compact_words if w]

    if not compact_words:
        return True

    text_l = cleaned.lower()
    if any(k in text_l for k in _INTRO_META_KEYWORDS):
        return True

    upper_chars = sum(1 for ch in text if ch.isalpha() and ch.isupper())
    alpha_chars = sum(1 for ch in text if ch.isalpha())
    upper_ratio = (upper_chars / alpha_chars) if alpha_chars else 0.0

    short_caps = (
        len(compact_words) <= 2
        and max((len(w) for w in compact_words), default=0) <= 4
        and upper_ratio >= 0.75
    )
    if short_caps:
        return True

    first_t = float(entry.get("first", 0.0))
    if first_t < 12.0 and len(compact_words) >= 4:
        mid = len(compact_words) // 2
        if compact_words[:mid] == compact_words[mid:]:
            return True

    all_title_case = (
        bool(compact_words)
        and all(
            token[:1].isupper() and (token[1:].islower() if len(token) > 1 else True)
            for token in text.split()
            if any(ch.isalpha() for ch in token)
        )
        and len(compact_words) <= 3
    )
    if all_title_case and first_t < 12.0:
        words_l = [w.lower() for w in compact_words]
        if all(w not in _LYRIC_FUNCTION_WORDS for w in words_l):
            return True

    if len(words) <= 2 and duration < 1.1 and "'" not in text and "-" not in text:
        return True

    return False


def _filter_intro_non_lyrics(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(entries) < 2:
        return entries

    anchor_idx = None
    candidates: list[int] = []
    for idx, ent in enumerate(entries):
        words = [w for w in ent.get("words", []) if str(w).strip()]
        duration = float(ent.get("last", 0.0)) - float(ent.get("first", 0.0))
        if len(words) >= 3 and duration >= 0.8 and not _is_intro_artifact(ent):
            candidates.append(idx)

    if candidates:
        later = [
            idx for idx in candidates if float(entries[idx].get("first", 0.0)) >= 10.0
        ]
        anchor_idx = later[0] if later else candidates[0]

    if anchor_idx is None or anchor_idx == 0:
        return entries

    first_t = float(entries[0].get("first", 0.0))
    anchor_t = float(entries[anchor_idx].get("first", 0.0))
    if anchor_t - first_t < 6.0:
        return entries

    kept: list[dict[str, Any]] = []
    for idx, ent in enumerate(entries):
        if idx < anchor_idx and _is_intro_artifact(ent):
            continue
        kept.append(ent)
    return kept


def reconstruct_lyrics_from_visuals(  # noqa: C901
    raw_frames: list[dict[str, Any]], visual_fps: float
) -> list[TargetLine]:
    """Group raw OCR words into logical lines and assign timing."""
    raw_frames = _filter_static_overlay_words(raw_frames)
    on_screen: Dict[str, Dict[str, Any]] = {}
    committed = []

    for frame in raw_frames:
        words = frame.get("words", [])
        current_norms = set()

        if words:
            # Sort by Y to process lines top-to-bottom
            words.sort(key=lambda w: w["y"])
            lines_in_frame = []

            # Group words into lines based on Y-proximity
            if words:
                curr = [words[0]]
                for i in range(1, len(words)):
                    if words[i]["y"] - curr[-1]["y"] < 20:
                        curr.append(words[i])
                    else:
                        lines_in_frame.append(curr)
                        curr = [words[i]]
                lines_in_frame.append(curr)

            for ln_w in lines_in_frame:
                ln_w.sort(key=lambda w: w["x"])
                line_tokens = normalize_ocr_tokens([str(w["text"]) for w in ln_w])
                if not line_tokens:
                    continue
                txt = normalize_ocr_line(" ".join(line_tokens))
                if not txt:
                    continue

                y_pos = int(sum(w["y"] for w in ln_w) / len(ln_w))
                # Create a key based on Y-bin and text content to track unique lines
                norm = f"y{y_pos // 30}_{normalize_text_basic(txt)}"
                current_norms.add(norm)

                if norm in on_screen:
                    on_screen[norm]["last"] = frame["time"]
                else:
                    lane = y_pos // 30
                    on_screen[norm] = {
                        "text": txt,
                        "words": line_tokens,
                        "first": frame["time"],
                        "last": frame["time"],
                        "y": y_pos,
                        "lane": lane,
                        "w_rois": [(w["x"], w["y"], w["w"], w["h"]) for w in ln_w],
                    }

        # Commit lines that have disappeared
        for nt in list(on_screen.keys()):
            # If line not seen in current frame and hasn't been seen for > 1.0s
            if nt not in current_norms and frame["time"] - on_screen[nt]["last"] > 1.0:
                committed.append(on_screen.pop(nt))

    # Commit remaining lines
    for ent in on_screen.values():
        committed.append(ent)

    # Deduplicate
    unique: list[dict[str, Any]] = []
    for ent in committed:
        is_dup = False
        for u in unique:
            # Text similarity check
            if text_similarity(ent["text"], u["text"]) > 0.9:
                # Spatial and Temporal proximity check
                if abs(ent["y"] - u["y"]) < 20 and abs(ent["first"] - u["first"]) < 2.0:
                    is_dup = True
                    break
        if not is_dup:
            unique.append(ent)

    # Sort: Primary by time (2.0s bins), Secondary by Y (top-to-bottom)
    # This keeps multi-line blocks together
    unique.sort(key=lambda x: (round(float(x["first"]) / 2.0) * 2.0, x["y"]))
    unique = _merge_short_same_lane_reentries(unique)
    unique = _suppress_short_duplicate_reentries(unique)
    unique = _filter_intro_non_lyrics(unique)

    out: list[TargetLine] = []
    for i, ent in enumerate(unique):
        s = snap(float(ent["first"]))
        # Determine end time based on next line start or duration
        if i + 1 < len(unique):
            nxt_s = snap(float(unique[i + 1]["first"]))
            # If next line starts soon (<3s), snap to it
            e = nxt_s if (nxt_s - s < 3.0) else snap(float(ent["last"]) + 2.0)
        else:
            e = snap(float(ent["last"]) + 2.0)

        out.append(
            TargetLine(
                line_index=i + 1,
                start=s,
                end=e,
                text=ent["text"],
                words=ent["words"],
                y=ent["y"],
                word_starts=None,
                word_ends=None,
                word_rois=ent["w_rois"],
                char_rois=None,
                visibility_start=float(ent["first"]),
                visibility_end=float(ent["last"]),
            )
        )
    return out
