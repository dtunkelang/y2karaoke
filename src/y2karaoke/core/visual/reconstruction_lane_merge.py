from __future__ import annotations

from typing import Any, Callable

from ..text_utils import text_similarity

EntryPredicate = Callable[[dict[str, Any]], bool]
EntryPairPredicate = Callable[[dict[str, Any], dict[str, Any]], bool]


def is_same_lane(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    lane_proximity_px: float,
) -> bool:
    return abs(float(a.get("y", 0.0)) - float(b.get("y", 0.0))) <= lane_proximity_px


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
            if is_short_refrain_entry(prev):
                continue
            if not is_same_lane(prev, ent):
                continue
            if (
                text_similarity(str(prev.get("text", "")), str(ent.get("text", "")))
                < 0.95
            ):
                continue
            prev_first = float(prev.get("first", 0.0))
            prev_last = float(prev.get("last", 0.0))
            cur_first = float(ent.get("first", 0.0))
            cur_last = float(ent.get("last", 0.0))
            overlap = min(prev_last, cur_last) - max(prev_first, cur_first)
            if overlap < 0.35:
                continue

            prev["first"] = min(prev_first, cur_first)
            prev["last"] = max(prev_last, cur_last)
            if len(ent.get("w_rois", [])) > len(prev.get("w_rois", [])):
                prev["w_rois"] = ent.get("w_rois", [])
            merged = True
            break

        if not merged:
            out.append(ent)
    return out
