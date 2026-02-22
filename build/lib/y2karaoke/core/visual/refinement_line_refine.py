"""Core per-line frame refinement helper."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from ..models import TargetLine

FrameTriplet = Tuple[float, Any, Any]


def refine_line_with_frames(
    ln: TargetLine,
    line_frames: List[FrameTriplet],
    *,
    word_fill_mask: Callable[[Any, Any], Any],
    detect_highlight_with_confidence: Callable[
        [List[dict[str, Any]]], Tuple[Optional[float], Optional[float], float]
    ],
    detect_line_highlight_with_confidence: Callable[
        [TargetLine, List[FrameTriplet], Any, Optional[float], bool],
        Tuple[Optional[float], Optional[float], float],
    ],
    assign_line_level_word_timings: Callable[
        [TargetLine, Optional[float], Optional[float], float], None
    ],
) -> None:
    c_bg_line = np.mean([np.mean(f[1], axis=(0, 1)) for f in line_frames[:10]], axis=0)

    new_starts: List[Optional[float]] = []
    new_ends: List[Optional[float]] = []
    new_confidences: List[Optional[float]] = []

    assert ln.word_rois is not None

    for wi in range(len(ln.words)):
        wx, wy, ww, wh = ln.word_rois[wi]
        word_vals = []
        for t, roi, roi_lab in line_frames:
            if wy + wh <= roi.shape[0] and wx + ww <= roi.shape[1]:
                word_roi = roi[wy : wy + wh, wx : wx + ww]
                mask = word_fill_mask(word_roi, c_bg_line)
                if np.sum(mask > 0) > 30:
                    lab = roi_lab[wy : wy + wh, wx : wx + ww]
                    word_vals.append(
                        {
                            "t": t,
                            "mask": mask,
                            "lab": lab,
                            "avg": lab[mask.astype(bool)].mean(axis=0),
                        }
                    )

        s, e, conf = detect_highlight_with_confidence(word_vals)
        new_starts.append(s)
        new_ends.append(e)
        new_confidences.append(conf)

    if not any(s is not None for s in new_starts):
        min_start = (
            float(ln.visibility_start) if ln.visibility_start is not None else None
        )
        line_s, line_e, line_conf = detect_line_highlight_with_confidence(
            ln,
            line_frames,
            c_bg_line,
            min_start,
            True,
        )
        if line_s is None and line_e is None:
            line_s, line_e, line_conf = detect_line_highlight_with_confidence(
                ln,
                line_frames,
                c_bg_line,
                min_start,
                False,
            )
        assign_line_level_word_timings(ln, line_s, line_e, line_conf)
        return

    ln.word_starts = new_starts
    ln.word_ends = new_ends
    ln.word_confidences = new_confidences
