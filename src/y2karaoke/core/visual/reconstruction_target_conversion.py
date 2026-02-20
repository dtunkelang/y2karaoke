"""Conversion logic from persistent visual lines to TargetLine models."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from ..models import TargetLine

SnapFn = Callable[[float], float]


def convert_persistent_lines_to_target_lines(
    persistent_lines: List[Dict[str, Any]],
    *,
    snap_fn: SnapFn,
) -> List[TargetLine]:
    """
    Converts a list of persistent line dictionaries into TargetLine objects,
    resolving final timing boundaries.
    """
    out: List[TargetLine] = []
    for i, ent in enumerate(persistent_lines):
        s = snap_fn(float(ent["first"]))
        if i + 1 < len(persistent_lines):
            nxt_s = snap_fn(float(persistent_lines[i + 1]["first"]))
            e = nxt_s if (nxt_s - s < 3.0) else snap_fn(float(ent["last"]) + 2.0)
        else:
            e = snap_fn(float(ent["last"]) + 2.0)

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
