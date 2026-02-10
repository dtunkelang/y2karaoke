"""Whisper-based mapping of LRC words to transcription timings.

Compatibility facade: implementation lives in focused submodules.
"""

from .whisper_mapping_helpers import (
    _SPEECH_BLOCK_GAP,
    _build_word_to_segment_index,
    _choose_segment_for_line,
    _collect_unused_words_in_window,
    _collect_unused_words_near_line,
    _dedupe_whisper_segments,
    _dedupe_whisper_words,
    _find_nearest_word_in_segment,
    _find_segment_for_time,
    _normalize_line_text,
    _segment_word_indices,
    _trim_whisper_transcription_by_lyrics,
    _word_match_score,
)
from .whisper_mapping_pipeline import (
    _assemble_mapped_line,
    _compute_gap_window,
    _fill_unmatched_gaps,
    _filter_and_order_candidates,
    _map_lrc_words_to_whisper,
    _match_assigned_words,
    _prepare_line_context,
    _register_word_match,
    _select_best_candidate,
)
from .whisper_mapping_post import (
    _build_word_assignments_from_phoneme_path,
    _enforce_monotonic_line_starts_whisper,
    _resolve_line_overlaps,
    _shift_repeated_lines_to_next_whisper,
)

__all__ = [
    "_SPEECH_BLOCK_GAP",
    "_dedupe_whisper_words",
    "_dedupe_whisper_segments",
    "_build_word_to_segment_index",
    "_find_segment_for_time",
    "_word_match_score",
    "_find_nearest_word_in_segment",
    "_normalize_line_text",
    "_trim_whisper_transcription_by_lyrics",
    "_choose_segment_for_line",
    "_segment_word_indices",
    "_collect_unused_words_near_line",
    "_collect_unused_words_in_window",
    "_register_word_match",
    "_select_best_candidate",
    "_filter_and_order_candidates",
    "_prepare_line_context",
    "_match_assigned_words",
    "_fill_unmatched_gaps",
    "_compute_gap_window",
    "_assemble_mapped_line",
    "_map_lrc_words_to_whisper",
    "_build_word_assignments_from_phoneme_path",
    "_shift_repeated_lines_to_next_whisper",
    "_enforce_monotonic_line_starts_whisper",
    "_resolve_line_overlaps",
]
