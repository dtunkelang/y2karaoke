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
            if dist_t < 1.5:
                # 1. Mangled OCR fragment vs stable line: allow vertical jitter (40px)
                #    if they are clearly intended to be the same (sim 0.4 - 0.9)
                if 0.4 <= sim <= 0.9 and dist_y < 40:
                    is_dup = True
                # 2. Near identical text (sim > 0.9): require tight proximity (20px)
                #    to keep distinct lanes separate (e.g. repeated 'Duh').
                elif sim > 0.9 and dist_y < 20:
                    is_dup = True
                # 3. Distinct noise: require tight proximity (20px)
                elif sim > 0.3 and dist_y < 20:
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

    # Primary sort by time (bucketed to 0.2s), secondary by Y (Top-to-Bottom)
    # The 0.2s bucket ensures lines that appear 'together' are ordered Top-then-Bottom
    # while keeping temporally distinct lines in their detected order.
    unique.sort(key=lambda x: (round(float(x["first"]) * 5.0) / 5.0, x["y"]))
    return unique
