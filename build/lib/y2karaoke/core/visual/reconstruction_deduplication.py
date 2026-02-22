"""Deduplication logic for persistent visual lines."""

from __future__ import annotations

from typing import Any, Dict, List

from ..text_utils import text_similarity


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

            # If close in time, check spatial proximity and similarity
            # Window of 5.0s is large enough to catch staggered 'shadow' repetitions
            if dist_t < 5.0:
                # 1. High similarity (sim > 0.7) in the SAME lane: collapse shadows
                if sim > 0.7 and dist_y < 30:
                    is_dup = True
                # 2. Mangled OCR fragment vs stable line: allow moderate vertical jitter (40px)
                elif 0.4 <= sim <= 0.7 and dist_y < 40:
                    is_dup = True

                if is_dup:
                    # Keep the 'better' version
                    ent_alpha = sum(1 for c in ent["text"] if c.isalpha())
                    u_alpha = sum(1 for c in u["text"] if c.isalpha())
                    ent_word_count = len(ent["words"])
                    u_word_count = len(u["words"])

                    # Prefer more words (fewer OCR fusions), or more letters
                    if ent_word_count > u_word_count or (
                        ent_word_count == u_word_count and ent_alpha > u_alpha
                    ):
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
