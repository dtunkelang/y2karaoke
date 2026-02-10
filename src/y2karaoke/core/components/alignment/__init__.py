"""Alignment component facade."""

from .timing_evaluator import (
    compare_sources,
    evaluate_timing,
    print_comparison_report,
    select_best_source,
)
from ..whisper.whisper_integration import (
    align_lrc_text_to_whisper_timings,
    correct_timing_with_whisper,
    transcribe_vocals,
)

__all__ = [
    "evaluate_timing",
    "compare_sources",
    "select_best_source",
    "print_comparison_report",
    "transcribe_vocals",
    "correct_timing_with_whisper",
    "align_lrc_text_to_whisper_timings",
]
