from __future__ import annotations

import os
from typing import Any, Callable

from ..text_utils import text_similarity

EntryPredicate = Callable[[dict[str, Any]], bool]
EntryPairPredicate = Callable[[dict[str, Any], dict[str, Any]], bool]


def _is_likely_nonvisible_tail_reentry(
    prev: dict[str, Any], ent: dict[str, Any], gap: float
) -> bool:
    prev_visible = bool(prev.get("visible_yet", False))
    ent_visible = bool(ent.get("visible_yet", False))
    return prev_visible and not ent_visible and gap >= 0.8


def suppress_short_duplicate_reentries(  # noqa: C901
    entries: list[dict[str, Any]],
    *,
    is_same_lane: EntryPairPredicate,
) -> list[dict[str, Any]]:
    if len(entries) < 2:
        return entries

    out: list[dict[str, Any]] = []
    for idx, ent in enumerate(entries):
        if bool(ent.get("_synthetic_repeat")):
            out.append(ent)
            continue
        duration = float(ent["last"]) - float(ent["first"])
        if duration > 1.2:
            out.append(ent)
            continue

        words = [str(w).strip() for w in ent.get("words", []) if str(w).strip()]
        if duration <= 0.35 and len(words) >= 3:
            has_near_stable_successor = False
            for nxt in entries[idx + 1 : idx + 10]:
                lead = float(nxt["first"]) - float(ent["first"])
                if lead < 0:
                    continue
                if lead > 4.0:
                    break
                if not is_same_lane(ent, nxt):
                    continue
                if text_similarity(ent["text"], nxt["text"]) < 0.9:
                    continue
                nxt_duration = float(nxt["last"]) - float(nxt["first"])
                if nxt_duration >= 0.8:
                    has_near_stable_successor = True
                    break
            if has_near_stable_successor:
                continue

        if duration <= 0.35 and len(words) >= 3:
            stable_prev: dict[str, Any] | None = None
            for prev in reversed(out[-10:]):
                prev_duration = float(prev["last"]) - float(prev["first"])
                if prev_duration < 1.0:
                    continue
                sim_prev = text_similarity(ent["text"], prev["text"])
                if (
                    sim_prev >= 0.35
                    and abs(float(prev["first"]) - float(ent["first"])) <= 0.6
                ):
                    stable_prev = prev
                    break
                if is_same_lane(ent, prev) and sim_prev >= 0.35:
                    stable_prev = prev
                    break
            if stable_prev is not None:
                stable_next: dict[str, Any] | None = None
                for nxt in entries[idx + 1 : idx + 10]:
                    lead = float(nxt["first"]) - float(ent["first"])
                    if lead < 0:
                        continue
                    if lead > 2.0:
                        break
                    if not is_same_lane(ent, nxt):
                        continue
                    nxt_duration = float(nxt["last"]) - float(nxt["first"])
                    if nxt_duration < 1.0:
                        continue
                    if text_similarity(ent["text"], nxt["text"]) >= 0.35:
                        stable_next = nxt
                        break
                if (
                    stable_next is not None
                    and text_similarity(stable_prev["text"], stable_next["text"]) >= 0.9
                    and text_similarity(ent["text"], stable_prev["text"]) < 0.9
                ):
                    short_anchor: dict[str, Any] | None = None
                    for cand in reversed(out[-8:]):
                        cand_duration = float(cand["last"]) - float(cand["first"])
                        cand_words = [
                            str(w).strip()
                            for w in cand.get("words", [])
                            if str(w).strip()
                        ]
                        if cand_duration > 0.35 or len(cand_words) > 2:
                            continue
                        if abs(float(cand["first"]) - float(ent["first"])) > 0.3:
                            continue
                        if text_similarity(cand["text"], stable_prev["text"]) >= 0.35:
                            continue
                        short_anchor = cand
                        break
                    if short_anchor is not None:
                        out.append(
                            {
                                "text": str(short_anchor["text"]),
                                "words": list(short_anchor.get("words", [])),
                                "first": float(ent["first"]),
                                "last": float(ent["last"]),
                                "y": int(ent.get("y", short_anchor.get("y", 0))),
                                "lane": int(
                                    ent.get(
                                        "lane",
                                        int(
                                            float(
                                                ent.get("y", short_anchor.get("y", 0.0))
                                            )
                                            // 30
                                        ),
                                    )
                                ),
                                "w_rois": list(short_anchor.get("w_rois", [])),
                            }
                        )
                    continue

        is_dup_reentry = False
        for prev in reversed(out[-12:]):
            time_gap = float(ent["first"]) - float(prev["first"])
            if time_gap < 0:
                continue
            if time_gap > 20.0:
                break
            if not is_same_lane(ent, prev):
                continue
            sim = text_similarity(ent["text"], prev["text"])
            prev_duration = float(prev["last"]) - float(prev["first"])
            if (
                duration <= 0.35
                and time_gap <= 8.0
                and prev_duration >= 0.6
                and sim >= 0.8
            ):
                is_dup_reentry = True
                break
            if sim < 0.9:
                continue
            if prev_duration >= 2.0 or prev_duration >= duration + 0.8:
                is_dup_reentry = True
                break

        if not is_dup_reentry:
            out.append(ent)
    return out


def merge_short_same_lane_reentries(
    entries: list[dict[str, Any]],
    *,
    is_same_lane: EntryPairPredicate,
    is_short_refrain_entry: EntryPredicate,
) -> list[dict[str, Any]]:
    """Merge short same-lane reentries caused by transient OCR drops."""
    if len(entries) < 2:
        return entries

    out: list[dict[str, Any]] = []

    for ent in entries:
        prev_idx = _find_recent_same_lane_duplicate_index(out, ent, is_same_lane)
        if prev_idx is None:
            out.append(ent)
            continue
        prev = out[prev_idx]
        gap = float(ent["first"]) - float(prev["last"])
        if gap < 0.0 or gap > 4.0:
            out.append(ent)
            continue

        mids = out[prev_idx + 1 :]
        if _should_merge_reentry(
            prev, ent, mids, gap, is_same_lane, is_short_refrain_entry
        ):
            _merge_entry_window(prev, ent)
            continue
        if _should_merge_short_token_reentry(prev, ent, mids, gap, is_same_lane):
            _merge_entry_window(prev, ent)
            continue

        out.append(ent)

    return out


def _find_recent_same_lane_duplicate_index(
    out: list[dict[str, Any]], ent: dict[str, Any], is_same_lane: EntryPairPredicate
) -> int | None:
    for idx in range(len(out) - 1, max(-1, len(out) - 13), -1):
        prev = out[idx]
        if text_similarity(prev["text"], ent["text"]) < 0.9:
            continue
        if not is_same_lane(prev, ent):
            continue
        return idx
    return None


def _has_lane_conflict(
    mids: list[dict[str, Any]], ent: dict[str, Any], is_same_lane: EntryPairPredicate
) -> bool:
    return any(
        is_same_lane(mid, ent) and text_similarity(mid["text"], ent["text"]) < 0.9
        for mid in mids
    )


def _cross_lane_same_text_exists(
    mids: list[dict[str, Any]], ent: dict[str, Any], is_same_lane: EntryPairPredicate
) -> bool:
    return any(
        not is_same_lane(mid, ent) and text_similarity(mid["text"], ent["text"]) >= 0.9
        for mid in mids
    )


def _ghost_guard_allows(prev: dict[str, Any], ent: dict[str, Any], gap: float) -> bool:
    if os.environ.get("Y2K_VISUAL_DISABLE_GHOST_REENTRY_GUARDS", "0") == "1":
        return True
    return not _is_likely_nonvisible_tail_reentry(prev, ent, gap)


def _should_merge_reentry(
    prev: dict[str, Any],
    ent: dict[str, Any],
    mids: list[dict[str, Any]],
    gap: float,
    is_same_lane: EntryPairPredicate,
    is_short_refrain_entry: EntryPredicate,
) -> bool:
    return (
        gap <= 1.5
        and not _has_lane_conflict(mids, ent, is_same_lane)
        and not (is_short_refrain_entry(prev) or is_short_refrain_entry(ent))
        and _ghost_guard_allows(prev, ent, gap)
    )


def _should_merge_short_token_reentry(
    prev: dict[str, Any],
    ent: dict[str, Any],
    mids: list[dict[str, Any]],
    gap: float,
    is_same_lane: EntryPairPredicate,
) -> bool:
    prev_duration = float(prev["last"]) - float(prev["first"])
    duration = float(ent["last"]) - float(ent["first"])
    allow_merge = prev_duration >= 1.0 or _cross_lane_same_text_exists(
        mids, ent, is_same_lane
    )
    ent_tokens = [t for t in ent.get("words", []) if str(t).strip()]
    prev_tokens = [t for t in prev.get("words", []) if str(t).strip()]
    return (
        duration <= 1.6
        and allow_merge
        and not _has_lane_conflict(mids, ent, is_same_lane)
        and len(ent_tokens) <= 2
        and len(prev_tokens) <= 2
        and _ghost_guard_allows(prev, ent, gap)
    )


def _merge_entry_window(prev: dict[str, Any], ent: dict[str, Any]) -> None:
    prev["last"] = max(float(prev["last"]), float(ent["last"]))
    if len(ent.get("w_rois", [])) > len(prev.get("w_rois", [])) and ent.get("w_rois"):
        prev["w_rois"] = ent["w_rois"]
