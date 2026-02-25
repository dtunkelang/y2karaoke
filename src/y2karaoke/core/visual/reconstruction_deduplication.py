"""Deduplication logic for persistent visual lines."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from ..text_utils import text_similarity


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

            if os.environ.get(
                "Y2K_VISUAL_DISABLE_GHOST_REENTRY_GUARDS", "0"
            ) != "1" and _is_likely_nonvisible_ghost_tail(ent, u, sim, dist_y):
                continue

            # 1. Standard deduplication: overlapping or very close in time (< 5.0s start-diff)
            if dist_t < 5.0:
                # High similarity in the SAME lane: collapse shadows
                if sim > 0.7 and dist_y < 30:
                    is_dup = True
                # Mangled OCR fragment vs stable line: allow moderate vertical jitter
                elif 0.6 <= sim <= 0.7 and dist_y < 40:
                    is_dup = True

            # 2. Ghost busting: lines separated by a gap (e.g. flickering static footer)
            # Require STRICT similarity to avoid merging different lines that share a lane.
            # Reduced gap threshold from 20.0 to 5.0 to avoid merging legitimate repetitions.
            elif gap_t < 5.0:
                if sim > 0.85 and dist_y < 30:
                    is_dup = True

            if is_dup:
                ent_dur = ent["last"] - ent["first"]
                u_dur = u["last"] - u["first"]
                ent_alpha = sum(1 for c in ent["text"] if c.isalpha())
                u_alpha = sum(1 for c in u["text"] if c.isalpha())
                ent_word_count = len(ent["words"])
                u_word_count = len(u["words"])

                # If one is significantly more stable (longer duration), prefer its text
                prefer_ent = False
                if ent_dur > u_dur + 1.0:
                    prefer_ent = True
                elif u_dur > ent_dur + 1.0:
                    prefer_ent = False
                else:
                    # Prefer more words (fewer OCR fusions), or more letters
                    if ent_word_count > u_word_count or (
                        ent_word_count == u_word_count and ent_alpha > u_alpha
                    ):
                        prefer_ent = True

                if prefer_ent:
                    u["text"] = ent["text"]
                    u["words"] = ent["words"]
                    u["w_rois"] = ent["w_rois"]

                u["first"] = min(u["first"], ent["first"])
                u["last"] = max(u["last"], ent["last"])
                break

        if not is_dup:
            unique.append(ent)

    # Primary sort by visibility onset time, but group lines that share significant overlap.
    # Logic: if two lines appear very close in time (within 2.0s), they belong to the
    # same visual block and should be ordered Top-then-Bottom by Y.
    # 2.0s is the 'sweet spot' for grouping fading-in Karafun pairs.
    unique.sort(key=lambda x: (round(float(x["first"]) / 2.0) * 2.0, x["y"]))
    return unique
