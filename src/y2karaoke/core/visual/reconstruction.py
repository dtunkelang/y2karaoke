from __future__ import annotations

from typing import Any, Dict

from ..models import TargetLine
from ..text_utils import (
    normalize_ocr_line,
    normalize_ocr_tokens,
    normalize_text_basic,
    text_similarity,
)
from .reconstruction_mirrored_cycles import (
    extrapolate_mirrored_lane_cycles as _extrapolate_mirrored_lane_cycles_impl,
)
from .reconstruction_mirrored_cycles import (
    is_candidate_for_mirrored_cycle as _is_candidate_for_mirrored_cycle_impl,
)
from .reconstruction_mirrored_cycles import (
    mirrored_cycle_candidate as _mirrored_cycle_candidate_impl,
)
from .reconstruction_overlay import _filter_static_overlay_words
from .word_segmentation import segment_line_tokens_by_visual_gaps

_LANE_PROXIMITY_PX = 18.0


def snap(value: float) -> float:
    # Assuming 0.05s snap from original tool
    return round(round(float(value) / 0.05) * 0.05, 3)


def _suppress_short_duplicate_reentries(  # noqa: C901
    entries: list[dict[str, Any]],
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

        # Suppress one-frame same-lane phrase fragments that are immediately
        # followed by a stable reappearance of the same text.
        words = [str(w).strip() for w in ent.get("words", []) if str(w).strip()]
        if duration <= 0.35 and len(words) >= 3:
            has_near_stable_successor = False
            for nxt in entries[idx + 1 : idx + 10]:
                lead = float(nxt["first"]) - float(ent["first"])
                if lead < 0:
                    continue
                if lead > 4.0:
                    break
                if not _is_same_lane(ent, nxt):
                    continue
                if text_similarity(ent["text"], nxt["text"]) < 0.9:
                    continue
                nxt_duration = float(nxt["last"]) - float(nxt["first"])
                if nxt_duration >= 0.8:
                    has_near_stable_successor = True
                    break
            if has_near_stable_successor:
                continue

        # Suppress one-frame distorted variants that sit between two stable
        # same-lane copies of (roughly) the same lyric line.
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
                if _is_same_lane(ent, prev) and sim_prev >= 0.35:
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
                    if not _is_same_lane(ent, nxt):
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
                    # When a distorted one-frame variant appears alongside a short
                    # one-frame refrain token (e.g. "Duh"), preserve the refrain
                    # repetition by remapping this ghost entry to that token.
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
            if not _is_same_lane(ent, prev):
                continue
            sim = text_similarity(ent["text"], prev["text"])
            prev_duration = float(prev["last"]) - float(prev["first"])
            # Ultra-short one-frame same-lane reentries are usually OCR ghosts.
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

        if prev_idx is not None:
            prev = out[prev_idx]
            prev_tokens = [t for t in prev.get("words", []) if str(t).strip()]
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
                is_short_refrain = _is_short_refrain_entry(
                    prev
                ) or _is_short_refrain_entry(ent)
                # OCR can briefly drop same-lane lines, then re-emit them as a new
                # entry. Stitch these continuation fragments for any line length.
                if (
                    continuation_split
                    and not has_lane_conflict
                    and not is_short_refrain
                ):
                    prev["last"] = max(float(prev["last"]), float(ent["last"]))
                    if len(ent.get("w_rois", [])) > len(
                        prev.get("w_rois", [])
                    ) and ent.get("w_rois"):
                        prev["w_rois"] = ent["w_rois"]
                    continue

                # For very short fragments, keep additional merge path but restrict
                # to short token runs to avoid collapsing real repeated long lines.
                if (
                    duration <= 1.6
                    and allow_merge
                    and not has_lane_conflict
                    and len(tokens) <= 2
                    and len(prev_tokens) <= 2
                ):
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


def _merge_overlapping_same_lane_duplicates(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge overlapping same-text epochs caused by lane-bin jitter."""
    if len(entries) < 2:
        return entries

    out: list[dict[str, Any]] = []
    for ent in entries:
        if not out or _is_short_refrain_entry(ent):
            out.append(ent)
            continue

        merged = False
        for prev in reversed(out[-10:]):
            if _is_short_refrain_entry(prev):
                continue
            if not _is_same_lane(prev, ent):
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


def _is_short_refrain_entry(entry: dict[str, Any]) -> bool:
    words = [
        normalize_text_basic(str(w)) for w in entry.get("words", []) if str(w).strip()
    ]
    words = [w for w in words if w]
    if len(words) < 3 or len(words) > 10:
        return False
    refrain_count = sum(1 for w in words if w in _SHORT_REFRAIN_TOKENS)
    return refrain_count / float(len(words)) >= 0.75


def _collapse_short_refrain_noise(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse overlapping same-lane short refrain OCR variants."""
    if len(entries) < 2:
        return entries

    out: list[dict[str, Any]] = []
    for ent in entries:
        if not _is_short_refrain_entry(ent):
            out.append(ent)
            continue

        merged = False
        for prev in reversed(out[-10:]):
            if not _is_short_refrain_entry(prev):
                continue
            if not _is_same_lane(prev, ent):
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


_INTRO_META_KEYWORDS = {
    "karaoke",
    "singking",
    "version",
    "official",
    "records",
    "universal",
    "kobalt",
    "publishing",
    "ltd",
    "mca",
    "oconnell",
    "connell",
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


def _suppress_bottom_fragment_families(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(entries) < 4:
        return entries

    ys = [float(ent.get("y", 0.0)) for ent in entries]
    y_min = min(ys)
    y_max = max(ys)
    if (y_max - y_min) < 80.0:
        return entries
    bottom_cut = y_min + 0.84 * (y_max - y_min)

    candidates: list[tuple[str, str]] = []
    for ent in entries:
        words = [str(w).strip() for w in ent.get("words", []) if str(w).strip()]
        if len(words) != 1:
            continue
        tok = words[0]
        compact = "".join(ch for ch in tok if ch.isalnum())
        if not compact or len(compact) > 4:
            continue
        upper_chars = sum(1 for ch in tok if ch.isalpha() and ch.isupper())
        alpha_chars = sum(1 for ch in tok if ch.isalpha())
        upper_ratio = (upper_chars / alpha_chars) if alpha_chars else 0.0
        if upper_ratio < 0.8:
            continue
        if float(ent.get("y", 0.0)) < bottom_cut:
            continue
        candidates.append((compact[0].lower(), compact.lower()))

    if not candidates:
        return entries

    counts: dict[str, int] = {}
    variants: dict[str, set[str]] = {}
    for initial, token in candidates:
        counts[initial] = counts.get(initial, 0) + 1
        variants.setdefault(initial, set()).add(token)

    noisy_initials = {
        initial
        for initial, count in counts.items()
        if count >= 4 and len(variants.get(initial, set())) >= 3
    }
    if not noisy_initials:
        return entries

    out: list[dict[str, Any]] = []
    for ent in entries:
        words = [str(w).strip() for w in ent.get("words", []) if str(w).strip()]
        if len(words) != 1:
            out.append(ent)
            continue
        tok = words[0]
        compact = "".join(ch for ch in tok if ch.isalnum())
        if not compact or len(compact) > 4:
            out.append(ent)
            continue
        initial = compact[0].lower()
        upper_chars = sum(1 for ch in tok if ch.isalpha() and ch.isupper())
        alpha_chars = sum(1 for ch in tok if ch.isalpha())
        upper_ratio = (upper_chars / alpha_chars) if alpha_chars else 0.0
        is_bottom = float(ent.get("y", 0.0)) >= bottom_cut
        if initial in noisy_initials and upper_ratio >= 0.8 and is_bottom:
            continue
        out.append(ent)
    return out


def _expand_overlapped_same_text_repetitions(  # noqa: C901
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Split long overlapped same-text lanes into repeated lyric occurrences.

    Some karaoke layouts keep identical lyric text in two lanes for long windows
    while refrain lines cycle between them. OCR sees stable text, so a single long
    entry can hide repeated lyric occurrences. This pass adds additional starts
    based on interleaved short-refrain cycles.
    """
    if len(entries) < 3:
        return entries

    out = list(entries)
    n = len(entries)
    for i in range(n):
        a = entries[i]
        if _is_short_refrain_entry(a):
            continue
        for j in range(i + 1, n):
            b = entries[j]
            if _is_short_refrain_entry(b):
                continue
            if text_similarity(a["text"], b["text"]) < 0.97:
                continue
            if _is_same_lane(a, b):
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
                if _is_short_refrain_entry(ent)
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
            # Infer additional repetition epochs when the anchor line stays on-screen
            # while companion lines transition quickly (common in outro loops where the
            # top line starts highlighting almost immediately after appearing).
            companion = [
                ent
                for ent in entries
                if overlap_start <= float(ent["first"]) <= overlap_end
                and text_similarity(str(ent.get("text", "")), str(anchor["text"])) < 0.9
                and not _is_short_refrain_entry(ent)
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
                    and abs(float(x["y"]) - float(anchor["y"])) <= _LANE_PROXIMITY_PX
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


def _extrapolate_mirrored_lane_cycles(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _extrapolate_mirrored_lane_cycles_impl(
        entries,
        lane_proximity_px=_LANE_PROXIMITY_PX,
        is_candidate_for_mirrored_cycle=_is_candidate_for_mirrored_cycle,
        mirrored_cycle_candidate=_mirrored_cycle_candidate,
    )


def _is_candidate_for_mirrored_cycle(entry: dict[str, Any]) -> bool:
    return _is_candidate_for_mirrored_cycle_impl(
        entry,
        is_short_refrain_entry=_is_short_refrain_entry,
    )


def _mirrored_cycle_candidate(
    a: dict[str, Any], b: dict[str, Any]
) -> tuple[float, dict[str, Any]] | None:
    return _mirrored_cycle_candidate_impl(
        a,
        b,
        is_candidate_for_mirrored_cycle=_is_candidate_for_mirrored_cycle,
        is_same_lane=_is_same_lane,
    )


def _split_persistent_line_epochs_from_context_transitions(  # noqa: C901
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add repeated epochs for long-lived lines when surrounding context cycles.

    Karaoke outros often keep a lead line visible while companion lines switch
    rapidly. If the lead line starts highlighting immediately on reappearance,
    OCR text alone can collapse multiple lyric occurrences into one long entry.
    This pass infers additional lead-line epochs from companion transition times.
    """
    if len(entries) < 3:
        return entries

    out = list(entries)
    for anchor in entries:
        words = [w for w in anchor.get("words", []) if str(w).strip()]
        if len(words) < 4:
            continue
        if _is_short_refrain_entry(anchor):
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
            <= _LANE_PROXIMITY_PX
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
                <= _LANE_PROXIMITY_PX
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
                line_tokens = segment_line_tokens_by_visual_gaps(ln_w)
                line_tokens = normalize_ocr_tokens(line_tokens)
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
    unique = _merge_overlapping_same_lane_duplicates(unique)
    unique = _merge_short_same_lane_reentries(unique)
    unique = _expand_overlapped_same_text_repetitions(unique)
    unique = _extrapolate_mirrored_lane_cycles(unique)
    unique = _split_persistent_line_epochs_from_context_transitions(unique)
    unique = _suppress_short_duplicate_reentries(unique)
    unique = _collapse_short_refrain_noise(unique)
    unique = _filter_intro_non_lyrics(unique)
    unique = _suppress_bottom_fragment_families(unique)

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
