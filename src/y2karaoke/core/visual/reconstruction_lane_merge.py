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

            start_a = a["first"]
            end_a = a["last"]
            start_b = b["first"]
            end_b = b["last"]

            overlap_start = max(start_a, start_b)
            overlap_end = min(end_a, end_b)
            overlap_duration = overlap_end - overlap_start

            if overlap_duration <= 0:
                # Also check for extreme proximity (gap < 0.2s)
                if abs(start_b - end_a) < 0.2 or abs(start_a - end_b) < 0.2:
                    overlap_duration = 0.1  # Trigger logic
                else:
                    continue

            dur_a = end_a - start_a
            dur_b = end_b - start_b
            bright_a = a.get("avg_brightness", 255.0)
            bright_b = b.get("avg_brightness", 255.0)

            # If one is much longer and brighter than the other
            # OR if they significantly overlap and one is clearly a fragment
            if (dur_a > dur_b * 2 and bright_a > bright_b * 1.2) or (
                overlap_duration / max(0.01, min(dur_a, dur_b)) > 0.5
                and (
                    (dur_a < 1.5 and dur_b > 3.0)
                    or (dur_b < 1.5 and dur_a > 3.0)
                    or (abs(bright_a - bright_b) > 40)
                )
            ):
                # Suppress the "inferior" one
                if bright_a > bright_b + 20 or dur_a > dur_b + 1.0:
                    keep[j] = False
                    # Extend A to cover B's time
                    a["first"] = min(a["first"], b["first"])
                    a["last"] = max(a["last"], b["last"])
                else:
                    keep[i] = False
                    b["first"] = min(b["first"], a["first"])
                    b["last"] = max(b["last"], a["last"])

    return [e for idx, e in enumerate(entries) if keep[idx]]
