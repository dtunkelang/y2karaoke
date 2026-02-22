from __future__ import annotations

from typing import Any, Callable

from ..text_utils import normalize_text_basic, text_similarity

EntryPredicate = Callable[[dict[str, Any]], bool]


def split_persistent_line_epochs_from_context_transitions(  # noqa: C901
    entries: list[dict[str, Any]],
    *,
    lane_proximity_px: float,
    is_short_refrain_entry: EntryPredicate,
) -> list[dict[str, Any]]:
    """Infer repeated epochs for long-lived lines from companion transitions."""
    if len(entries) < 3:
        return entries

    out = list(entries)
    for anchor in entries:
        words = [w for w in anchor.get("words", []) if str(w).strip()]
        if len(words) < 4:
            continue
        if is_short_refrain_entry(anchor):
            continue
        a_first = float(anchor.get("first", 0.0))
        a_last = float(anchor.get("last", 0.0))
        a_span = a_last - a_first
        if a_span < 5.5:
            continue

        companion = [
            ent
            for ent in entries
            if ent is not anchor
            and text_similarity(str(ent.get("text", "")), str(anchor.get("text", "")))
            < 0.85
            and abs(float(ent.get("y", 0.0)) - float(anchor.get("y", 0.0))) > 16.0
            and (a_first - 0.2) <= float(ent.get("first", 0.0)) <= (a_last + 0.2)
        ]
        if len(companion) < 2:
            continue
        companion.sort(key=lambda x: float(x.get("first", 0.0)))
        family_keys = {
            normalize_text_basic(str(ent.get("text", ""))).split(" ")[0]
            for ent in companion
            if normalize_text_basic(str(ent.get("text", ""))).strip()
        }
        if len(family_keys) < 2:
            continue

        min_start = a_first + 2.0
        max_start = a_last - 0.8
        existing_same_text = [
            ent
            for ent in entries
            if text_similarity(str(ent.get("text", "")), str(anchor.get("text", "")))
            >= 0.97
            and abs(float(ent.get("y", 0.0)) - float(anchor.get("y", 0.0)))
            <= lane_proximity_px
        ]
        if len(existing_same_text) >= 4:
            continue

        def _family_key(ent: dict[str, Any]) -> str:
            norm = normalize_text_basic(str(ent.get("text", "")))
            toks = [t for t in norm.split(" ") if t][:3]
            return " ".join(toks)

        starts: list[float] = []
        prev_family: str | None = None
        prev_family_t: float | None = None
        for ent in companion:
            t = float(ent.get("first", 0.0))
            if t <= min_start or t >= max_start:
                continue
            family = _family_key(ent)
            if not family:
                continue
            if prev_family is None:
                prev_family = family
                prev_family_t = t
                continue
            if family == prev_family:
                prev_family_t = t
                continue
            if prev_family_t is not None and (t - prev_family_t) < 1.1:
                prev_family = family
                prev_family_t = t
                continue
            if starts and (t - starts[-1]) < 2.2:
                prev_family = family
                prev_family_t = t
                continue
            starts.append(t)
            prev_family = family
            prev_family_t = t
        if not starts:
            continue

        for t in starts[:2]:
            already_present = any(
                text_similarity(str(x.get("text", "")), str(anchor.get("text", "")))
                >= 0.97
                and abs(float(x.get("y", 0.0)) - float(anchor.get("y", 0.0)))
                <= lane_proximity_px
                and abs(float(x.get("first", 0.0)) - t) <= 1.2
                for x in out
            )
            if already_present:
                continue
            out.append(
                {
                    "text": str(anchor.get("text", "")),
                    "words": list(anchor.get("words", [])),
                    "first": t,
                    "last": min(a_last, t + 1.2),
                    "y": int(anchor.get("y", 0)),
                    "lane": int(
                        anchor.get("lane", int(float(anchor.get("y", 0.0))) // 30)
                    ),
                    "w_rois": list(anchor.get("w_rois", [])),
                    "_synthetic_repeat": True,
                }
            )

    out.sort(key=lambda x: (float(x.get("first", 0.0)), float(x.get("y", 0.0))))
    return out
