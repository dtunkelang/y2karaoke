from __future__ import annotations

from typing import Any, Callable

from ..text_utils import text_similarity

SimilarityFn = Callable[[str, str], float]
EntryPredicate = Callable[[dict[str, Any]], bool]
EntryPairPredicate = Callable[[dict[str, Any], dict[str, Any]], bool]


def extrapolate_mirrored_lane_cycles(
    entries: list[dict[str, Any]],
    *,
    lane_proximity_px: float,
    is_candidate_for_mirrored_cycle: EntryPredicate,
    mirrored_cycle_candidate: Callable[
        [dict[str, Any], dict[str, Any]], tuple[float, dict[str, Any]] | None
    ],
    similarity_fn: SimilarityFn = text_similarity,
) -> list[dict[str, Any]]:
    """Infer one additional cycle for long mirrored same-text lane pairs."""
    if len(entries) < 4:
        return entries

    out = list(entries)
    n = len(entries)
    for i in range(n):
        a = entries[i]
        if not is_candidate_for_mirrored_cycle(a):
            continue

        for j in range(i + 1, n):
            b = entries[j]
            candidate = mirrored_cycle_candidate(a, b)
            if candidate is None:
                continue
            extra_t, anchor = candidate

            already_present = any(
                similarity_fn(str(x.get("text", "")), str(anchor.get("text", "")))
                >= 0.97
                and abs(float(x.get("y", 0.0)) - float(anchor.get("y", 0.0)))
                <= lane_proximity_px
                and abs(float(x.get("first", 0.0)) - extra_t) <= 1.2
                for x in out
            )
            if already_present:
                continue

            out.append(
                {
                    "text": str(anchor.get("text", "")),
                    "words": list(anchor.get("words", [])),
                    "first": extra_t,
                    "last": min(float(anchor.get("last", 0.0)), extra_t + 1.2),
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


def is_candidate_for_mirrored_cycle(
    entry: dict[str, Any],
    *,
    is_short_refrain_entry: EntryPredicate,
) -> bool:
    if is_short_refrain_entry(entry):
        return False
    first = float(entry.get("first", 0.0))
    last = float(entry.get("last", 0.0))
    return (last - first) >= 6.0


def mirrored_cycle_candidate(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    is_candidate_for_mirrored_cycle: EntryPredicate,
    is_same_lane: EntryPairPredicate,
    similarity_fn: SimilarityFn = text_similarity,
) -> tuple[float, dict[str, Any]] | None:
    if not is_candidate_for_mirrored_cycle(b):
        return None
    if is_same_lane(a, b):
        return None
    if similarity_fn(str(a.get("text", "")), str(b.get("text", ""))) < 0.97:
        return None

    a_first = float(a.get("first", 0.0))
    a_last = float(a.get("last", 0.0))
    b_first = float(b.get("first", 0.0))
    b_last = float(b.get("last", 0.0))
    overlap_end = min(a_last, b_last)
    overlap_span = overlap_end - max(a_first, b_first)
    if overlap_span < 7.0:
        return None

    gap = abs(a_first - b_first)
    if gap < 4.0 or gap > 10.5:
        return None
    if overlap_span < max(6.5, gap - 0.5):
        return None

    seed = max(a_first, b_first)
    extra_t = seed + gap
    if extra_t >= (overlap_end - 1.0):
        return None
    if extra_t <= (seed + 2.0):
        return None
    anchor = a if a_first >= b_first else b
    return extra_t, anchor
