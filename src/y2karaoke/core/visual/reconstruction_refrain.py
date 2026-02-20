from __future__ import annotations

from typing import Any, Callable

from ..text_utils import normalize_text_basic, text_similarity

EntryPredicate = Callable[[dict[str, Any]], bool]
EntryPairPredicate = Callable[[dict[str, Any], dict[str, Any]], bool]

_SHORT_REFRAIN_TOKENS = {
    "oh",
    "ooh",
    "woah",
    "i",
    "im",
    "i'm",
    "yeah",
    "uh",
}


def is_short_refrain_entry(entry: dict[str, Any]) -> bool:
    words = [
        normalize_text_basic(str(w)) for w in entry.get("words", []) if str(w).strip()
    ]
    words = [w for w in words if w]
    if len(words) < 3 or len(words) > 10:
        return False
    refrain_count = sum(1 for w in words if w in _SHORT_REFRAIN_TOKENS)
    return refrain_count / float(len(words)) >= 0.75


def collapse_short_refrain_noise(
    entries: list[dict[str, Any]],
    *,
    is_short_refrain_entry: EntryPredicate,
    is_same_lane: EntryPairPredicate,
) -> list[dict[str, Any]]:
    """Collapse overlapping same-lane short refrain OCR variants."""
    if len(entries) < 2:
        return entries

    out: list[dict[str, Any]] = []
    for ent in entries:
        if not is_short_refrain_entry(ent):
            out.append(ent)
            continue

        merged = False
        for prev in reversed(out[-10:]):
            if not is_short_refrain_entry(prev):
                continue
            if not is_same_lane(prev, ent):
                continue
            start_gap = abs(
                float(ent.get("first", 0.0)) - float(prev.get("first", 0.0))
            )
            end_gap = float(ent.get("first", 0.0)) - float(prev.get("last", 0.0))
            sim = text_similarity(str(prev.get("text", "")), str(ent.get("text", "")))
            if start_gap <= 2.2 and end_gap <= 1.8 and sim >= 0.45:
                prev["last"] = max(
                    float(prev.get("last", 0.0)), float(ent.get("last", 0.0))
                )
                if len(ent.get("w_rois", [])) > len(prev.get("w_rois", [])):
                    prev["w_rois"] = ent.get("w_rois", [])
                merged = True
                break
        if not merged:
            out.append(ent)
    return out
