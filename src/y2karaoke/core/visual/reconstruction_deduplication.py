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
            if text_similarity(ent["text"], u["text"]) > 0.9:
                if abs(ent["y"] - u["y"]) < 20 and abs(ent["first"] - u["first"]) < 0.5:
                    is_dup = True
                    break
        if not is_dup:
            unique.append(ent)

    unique.sort(key=lambda x: (x["first"], x["y"]))
    return unique
