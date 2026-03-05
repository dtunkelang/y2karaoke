from __future__ import annotations

import os
from typing import Any, Callable

from ..text_utils import normalize_text_basic, text_similarity

EntryPredicate = Callable[[dict[str, Any]], bool]
EntryPairPredicate = Callable[[dict[str, Any], dict[str, Any]], bool]


def is_same_lane(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    lane_proximity_px: float,
) -> bool:
    return abs(float(a.get("y", 0.0)) - float(b.get("y", 0.0))) <= lane_proximity_px


def _is_merge_candidate(
    prev: dict[str, Any],
    ent: dict[str, Any],
    *,
    is_short_refrain_entry: EntryPredicate,
    is_same_lane: EntryPairPredicate,
) -> bool:
    if is_short_refrain_entry(prev) or not is_same_lane(prev, ent):
        return False
    sim = text_similarity(str(prev.get("text", "")), str(ent.get("text", "")))
    if sim < 0.95:
        return False
    prev_first = float(prev.get("first", 0.0))
    prev_last = float(prev.get("last", 0.0))
    cur_first = float(ent.get("first", 0.0))
    cur_last = float(ent.get("last", 0.0))
    overlap = min(prev_last, cur_last) - max(prev_first, cur_first)
    return overlap >= 0.35


def _merge_duplicate_entry(prev: dict[str, Any], ent: dict[str, Any]) -> None:
    prev_first = float(prev.get("first", 0.0))
    prev_last = float(prev.get("last", 0.0))
    cur_first = float(ent.get("first", 0.0))
    cur_last = float(ent.get("last", 0.0))
    prev["first"] = min(prev_first, cur_first)
    prev["last"] = max(prev_last, cur_last)
    if len(ent.get("w_rois", [])) > len(prev.get("w_rois", [])):
        prev["w_rois"] = ent.get("w_rois", [])


def merge_overlapping_same_lane_duplicates(
    entries: list[dict[str, Any]],
    *,
    is_short_refrain_entry: EntryPredicate,
    is_same_lane: EntryPairPredicate,
) -> list[dict[str, Any]]:
    """Merge overlapping same-text epochs caused by lane-bin jitter."""
    if len(entries) < 2:
        return entries

    out: list[dict[str, Any]] = []
    for ent in entries:
        if not out or is_short_refrain_entry(ent):
            out.append(ent)
            continue

        merged = False
        for prev in reversed(out[-10:]):
            if not _is_merge_candidate(
                prev,
                ent,
                is_short_refrain_entry=is_short_refrain_entry,
                is_same_lane=is_same_lane,
            ):
                continue
            _merge_duplicate_entry(prev, ent)
            merged = True
            break

        if not merged:
            out.append(ent)
    return out


def _overlap_or_close_proximity(a: dict[str, Any], b: dict[str, Any]) -> float:
    start_a = float(a["first"])
    end_a = float(a["last"])
    start_b = float(b["first"])
    end_b = float(b["last"])
    overlap_start = max(start_a, start_b)
    overlap_end = min(end_a, end_b)
    overlap_duration = overlap_end - overlap_start
    if overlap_duration > 0:
        return overlap_duration
    if abs(start_b - end_a) < 0.2 or abs(start_a - end_b) < 0.2:
        return 0.1
    return 0.0


def _should_merge_fade_pair(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    overlap_duration: float,
) -> bool:
    dur_a = float(a["last"]) - float(a["first"])
    dur_b = float(b["last"]) - float(b["first"])
    bright_a = float(a.get("avg_brightness", 255.0))
    bright_b = float(b.get("avg_brightness", 255.0))
    long_bright_domination = dur_a > dur_b * 2 and bright_a > bright_b * 1.2
    fragment_overlap = overlap_duration / max(0.01, min(dur_a, dur_b)) > 0.5 and (
        (dur_a < 1.5 and dur_b > 3.0)
        or (dur_b < 1.5 and dur_a > 3.0)
        or (abs(bright_a - bright_b) > 40)
    )
    return long_bright_domination or fragment_overlap


def _suppress_inferior_fade_entry(
    a: dict[str, Any],
    b: dict[str, Any],
    keep: list[bool],
    *,
    i: int,
    j: int,
) -> None:
    dur_a = float(a["last"]) - float(a["first"])
    dur_b = float(b["last"]) - float(b["first"])
    bright_a = float(a.get("avg_brightness", 255.0))
    bright_b = float(b.get("avg_brightness", 255.0))
    if bright_a > bright_b + 20 or dur_a > dur_b + 1.0:
        keep[j] = False
        a["first"] = min(a["first"], b["first"])
        a["last"] = max(a["last"], b["last"])
    else:
        keep[i] = False
        b["first"] = min(b["first"], a["first"])
        b["last"] = max(b["last"], a["last"])


def merge_dim_fade_in_fragments(
    entries: list[dict[str, Any]],
    *,
    is_same_lane: EntryPairPredicate,
) -> list[dict[str, Any]]:
    """Merge short, dim segments into longer overlapping segments in the same lane."""
    if len(entries) < 2:
        return entries

    keep = [True] * len(entries)
    for i, a in enumerate(entries):
        for j, b in enumerate(entries):
            if i == j or not keep[i] or not keep[j]:
                continue

            if not is_same_lane(a, b):
                continue

            if os.environ.get(
                "Y2K_VISUAL_DISABLE_FADE_FRAGMENT_TEXT_GUARD", "0"
            ) != "1" and not _looks_like_same_line_fade_fragment(a, b):
                continue

            overlap_duration = _overlap_or_close_proximity(a, b)
            if overlap_duration <= 0:
                continue
            if not _should_merge_fade_pair(a, b, overlap_duration=overlap_duration):
                continue
            _suppress_inferior_fade_entry(a, b, keep, i=i, j=j)

    return [e for idx, e in enumerate(entries) if keep[idx]]


def _looks_like_same_line_fade_fragment(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ta = str(a.get("text", ""))
    tb = str(b.get("text", ""))
    sim = text_similarity(ta, tb)
    if sim >= 0.72:
        return True

    toks_a = [t for t in normalize_text_basic(ta).split() if t]
    toks_b = [t for t in normalize_text_basic(tb).split() if t]
    if not toks_a or not toks_b:
        return False
    set_a = set(toks_a)
    set_b = set(toks_b)
    overlap = len(set_a & set_b) / float(max(1, min(len(set_a), len(set_b))))
    if overlap < 0.75:
        return False

    # Require a strong containment-like relationship for low string-similarity merges.
    shorter, longer = (
        (toks_a, toks_b) if len(toks_a) <= len(toks_b) else (toks_b, toks_a)
    )
    shorter_join = " ".join(shorter)
    longer_join = " ".join(longer)
    return shorter_join in longer_join
