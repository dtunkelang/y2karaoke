from __future__ import annotations

from typing import Any, Callable

from ..text_utils import text_similarity

EntryPredicate = Callable[[dict[str, Any]], bool]
EntryPairPredicate = Callable[[dict[str, Any], dict[str, Any]], bool]


def expand_overlapped_same_text_repetitions(  # noqa: C901
    entries: list[dict[str, Any]],
    *,
    lane_proximity_px: float,
    is_short_refrain_entry: EntryPredicate,
    is_same_lane: EntryPairPredicate,
) -> list[dict[str, Any]]:
    """Split long overlapped same-text lanes into repeated lyric occurrences."""
    if len(entries) < 3:
        return entries

    out = list(entries)
    n = len(entries)
    for i in range(n):
        a = entries[i]
        if is_short_refrain_entry(a):
            continue
        for j in range(i + 1, n):
            b = entries[j]
            if is_short_refrain_entry(b):
                continue
            if text_similarity(a["text"], b["text"]) < 0.97:
                continue
            if is_same_lane(a, b):
                continue

            a_first, a_last = float(a["first"]), float(a["last"])
            b_first, b_last = float(b["first"]), float(b["last"])
            overlap_start = max(a_first, b_first)
            overlap_end = min(a_last, b_last)
            if overlap_end - overlap_start < 3.0:
                continue

            da = a_last - a_first
            db = b_last - b_first
            if max(da, db) < 8.0:
                continue

            anchor = a if da >= db else b
            other = b if anchor is a else a
            if float(other["first"]) - float(anchor["first"]) < 4.0:
                continue

            y_lo = min(float(a["y"]), float(b["y"]))
            y_hi = max(float(a["y"]), float(b["y"]))
            refrain = [
                ent
                for ent in entries
                if is_short_refrain_entry(ent)
                and overlap_start <= float(ent["first"]) <= overlap_end
            ]
            if len(refrain) < 3:
                continue
            mid_refrain = [
                ent
                for ent in refrain
                if (y_lo + 10.0) <= float(ent["y"]) <= (y_hi - 10.0)
            ]
            refrain_source = mid_refrain if len(mid_refrain) >= 2 else refrain

            starts: list[float] = []
            min_start = float(other["first"]) + 2.2
            for ent in sorted(refrain_source, key=lambda x: float(x["first"])):
                t = float(ent["first"])
                if t <= min_start:
                    continue
                if starts and (t - starts[-1]) < 3.0:
                    continue
                starts.append(t)
            if len(starts) < 2:
                long_refrain = max(
                    refrain_source,
                    key=lambda x: float(x["last"]) - float(x["first"]),
                )
                long_span = float(long_refrain["last"]) - float(long_refrain["first"])
                if long_span >= 6.0:
                    t = max(min_start, float(long_refrain["first"]) + 2.2)
                    end_limit = min(
                        overlap_end - 1.0, float(long_refrain["last"]) - 1.0
                    )
                    while t <= end_limit:
                        if not starts or (t - starts[-1]) >= 3.0:
                            starts.append(t)
                        t += 4.0

            companion = [
                ent
                for ent in entries
                if overlap_start <= float(ent["first"]) <= overlap_end
                and text_similarity(str(ent.get("text", "")), str(anchor["text"])) < 0.9
                and not is_short_refrain_entry(ent)
            ]
            companion.sort(key=lambda x: float(x["first"]))
            if companion:
                prev_t = None
                for ent in companion:
                    t = float(ent["first"])
                    if t <= min_start:
                        continue
                    if prev_t is not None and (t - prev_t) < 1.4:
                        prev_t = t
                        continue
                    prev_t = t
                    if starts and any(abs(t - s) < 1.6 for s in starts):
                        continue
                    starts.append(t)
            starts.sort()
            if not starts:
                continue

            for t in starts[:6]:
                already_present = any(
                    text_similarity(x["text"], anchor["text"]) >= 0.97
                    and abs(float(x["y"]) - float(anchor["y"])) <= lane_proximity_px
                    and abs(float(x["first"]) - t) <= 1.2
                    for x in out
                )
                if already_present:
                    continue
                out.append(
                    {
                        "text": anchor["text"],
                        "words": list(anchor.get("words", [])),
                        "first": t,
                        "last": min(float(anchor["last"]), t + 1.2),
                        "y": int(anchor["y"]),
                        "lane": int(anchor.get("lane", int(anchor["y"]) // 30)),
                        "w_rois": list(anchor.get("w_rois", [])),
                        "_synthetic_repeat": True,
                    }
                )

    out.sort(key=lambda x: (float(x["first"]), float(x["y"])))
    return out
