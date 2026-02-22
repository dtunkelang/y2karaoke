"""Alignment subsystem facade.

Public orchestration entrypoints for timing evaluation and Whisper-based
alignment.
"""

from ...core.components.alignment import (
    align_lrc_text_to_whisper_timings,
    compare_sources,
    correct_timing_with_whisper,
    evaluate_timing,
    print_comparison_report,
    select_best_source,
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
