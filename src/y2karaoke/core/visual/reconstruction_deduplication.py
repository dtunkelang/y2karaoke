"""Deduplication logic for persistent visual lines."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from ..text_utils import text_similarity


def _ghost_reentry_guards_enabled() -> bool:
    return os.environ.get("Y2K_VISUAL_DISABLE_GHOST_REENTRY_GUARDS", "0") != "1"


def _is_likely_nonvisible_ghost_tail(
    ent: Dict[str, Any], u: Dict[str, Any], sim: float, dist_y: float
) -> bool:
    """Avoid merging later ghost-only continuations into earlier visible lines."""
    ent_visible = bool(ent.get("visible_yet", False))
    u_visible = bool(u.get("visible_yet", False))
    if ent_visible or not u_visible:
        return False
    if sim <= 0.8 or dist_y >= 30:
        return False
    # Later non-visible continuations often come from dim OCR persistence through
    # instrumental sections and should not extend the earlier visible line.
    return float(ent["first"]) >= float(u["last"]) + 0.8


def _is_duplicate_candidate(
    *, sim: float, dist_y: float, dist_t: float, gap_t: float
) -> bool:
    if dist_t < 5.0:
        if sim > 0.7 and dist_y < 30:
            return True
        if 0.6 <= sim <= 0.7 and dist_y < 40:
            return True
        return False
    if gap_t < 5.0:
        return sim > 0.85 and dist_y < 30
    return False


def _prefer_new_entity(ent: Dict[str, Any], u: Dict[str, Any]) -> bool:
    ent_dur = ent["last"] - ent["first"]
    u_dur = u["last"] - u["first"]
    ent_alpha = sum(1 for c in ent["text"] if c.isalpha())
    u_alpha = sum(1 for c in u["text"] if c.isalpha())
    ent_word_count = len(ent["words"])
    u_word_count = len(u["words"])

    if ent_dur > u_dur + 1.0:
        return True
    if u_dur > ent_dur + 1.0:
        return False
    return ent_word_count > u_word_count or (
        ent_word_count == u_word_count and ent_alpha > u_alpha
    )


def _merge_duplicate_into_unique(ent: Dict[str, Any], u: Dict[str, Any]) -> None:
    if _prefer_new_entity(ent, u):
        u["text"] = ent["text"]
        u["words"] = ent["words"]
        u["w_rois"] = ent["w_rois"]
    u["first"] = min(u["first"], ent["first"])
    u["last"] = max(u["last"], ent["last"])


def deduplicate_persistent_lines(
    committed_lines: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Filters out duplicate persistent lines that are spatially and temporally close
    with very similar text content.
    """
    unique: List[Dict[str, Any]] = []
    for ent in committed_lines:
        is_dup = False
        for u in unique:
            sim = text_similarity(ent["text"], u["text"])
            dist_y = abs(ent["y"] - u["y"])
            dist_t = abs(ent["first"] - u["first"])
            gap_t = max(0.0, ent["first"] - u["last"])

            if _ghost_reentry_guards_enabled() and _is_likely_nonvisible_ghost_tail(
                ent, u, sim, dist_y
            ):
                continue

            is_dup = _is_duplicate_candidate(
                sim=sim, dist_y=dist_y, dist_t=dist_t, gap_t=gap_t
            )
            if is_dup:
                _merge_duplicate_into_unique(ent, u)
                break

        if not is_dup:
            unique.append(ent)

    # Primary sort by visibility onset time, but group lines that share significant overlap.
    # Logic: if two lines appear very close in time (within 2.0s), they belong to the
    # same visual block and should be ordered Top-then-Bottom by Y.
    # 2.0s is the 'sweet spot' for grouping fading-in Karafun pairs.
    unique.sort(key=lambda x: (round(float(x["first"]) / 2.0) * 2.0, x["y"]))
    return unique
