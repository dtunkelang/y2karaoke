"""Parallel experimental segment assignment entry point.

This module exists to isolate larger redesigns of stalled-run candidate
generation, ownership, and distribution from the default live path.
"""

from __future__ import annotations

import os
from typing import Dict, List

from ..alignment import timing_models
from . import whisper_blocks

_EXPERIMENTAL_ASSIGNER_MODE = "parallel_experimental"


def _segment_assign_pipeline_mode() -> str:
    return (
        os.getenv("Y2K_WHISPER_SEGMENT_ASSIGN_PIPELINE", "default").strip() or "default"
    )


def build_segment_text_overlap_assignments(
    lrc_words: List[Dict],
    all_words: List[timing_models.TranscriptionWord],
    segments: List[timing_models.TranscriptionSegment],
) -> Dict[int, List[int]]:
    """Dispatch between the stable and experimental segment assigners."""
    mode = _segment_assign_pipeline_mode()
    if mode == _EXPERIMENTAL_ASSIGNER_MODE:
        return _build_segment_text_overlap_assignments_experimental(
            lrc_words=lrc_words,
            all_words=all_words,
            segments=segments,
        )
    return whisper_blocks._build_segment_text_overlap_assignments(
        lrc_words=lrc_words,
        all_words=all_words,
        segments=segments,
    )


def _build_segment_text_overlap_assignments_experimental(
    *,
    lrc_words: List[Dict],
    all_words: List[timing_models.TranscriptionWord],
    segments: List[timing_models.TranscriptionSegment],
) -> Dict[int, List[int]]:
    """Experimental parallel assigner scaffold.

    The initial version intentionally preserves current behavior. The purpose
    is to give stalled-run redesign work a dedicated implementation surface
    without perturbing the default path.
    """
    return whisper_blocks._build_segment_text_overlap_assignments(
        lrc_words=lrc_words,
        all_words=all_words,
        segments=segments,
    )
