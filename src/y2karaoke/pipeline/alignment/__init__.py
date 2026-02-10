"""Alignment subsystem facade.

Public orchestration entrypoints for timing evaluation and Whisper-based
alignment.
"""

from ...core.timing_evaluator import (
    evaluate_timing,
    compare_sources,
    select_best_source,
    print_comparison_report,
)
from ...core.whisper_integration import (
    transcribe_vocals,
    correct_timing_with_whisper,
    align_lrc_text_to_whisper_timings,
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
