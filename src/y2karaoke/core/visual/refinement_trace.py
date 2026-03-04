"""Trace helpers for visual refinement pipelines."""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

from ..models import TargetLine

logger = logging.getLogger(__name__)


def trace_refinement_snapshot(
    label: str, jobs: List[Tuple[TargetLine, float, float]]
) -> None:
    if os.environ.get("Y2K_VISUAL_REFINE_TRACE", "0") != "1":
        return
    lines = [ln for ln, _, _ in jobs]
    preview = []
    for ln in lines[:12]:
        if ln.word_starts and ln.word_starts[0] is not None:
            s = float(ln.word_starts[0])
        else:
            s = float(ln.start or 0.0)
        if ln.word_ends and ln.word_ends[-1] is not None:
            e = float(ln.word_ends[-1])
        elif ln.end is not None:
            e = float(ln.end)
        else:
            e = s
        preview.append(
            {
                "i": ln.line_index,
                "s": round(s, 2),
                "e": round(e, 2),
                "y": round(float(ln.y), 1),
                "vs": (
                    round(float(ln.visibility_start), 2)
                    if ln.visibility_start is not None
                    else None
                ),
                "ve": (
                    round(float(ln.visibility_end), 2)
                    if ln.visibility_end is not None
                    else None
                ),
                "t": " ".join(ln.words[:6]),
            }
        )
    logger.info("REFINE_TRACE %s first12=%s", label, preview)
