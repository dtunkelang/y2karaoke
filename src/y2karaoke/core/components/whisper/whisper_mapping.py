"""Whisper mapping compatibility facade.

Implementation now lives in focused submodules. Keep legacy attribute access
working lazily so compatibility callers do not force eager imports.
"""

from __future__ import annotations

from importlib import import_module

__all__: list[str] = []

_LEGACY_EXPORTS = {
    "_SPEECH_BLOCK_GAP": (
        "y2karaoke.core.components.whisper.whisper_mapping_helpers",
        "_SPEECH_BLOCK_GAP",
    ),
    "_assemble_mapped_line": (
        "y2karaoke.core.components.whisper.whisper_mapping_pipeline",
        "_assemble_mapped_line",
    ),
    "_build_word_assignments_from_phoneme_path": (
        "y2karaoke.core.components.whisper.whisper_mapping_post",
        "_build_word_assignments_from_phoneme_path",
    ),
    "_build_word_to_segment_index": (
        "y2karaoke.core.components.whisper.whisper_mapping_helpers",
        "_build_word_to_segment_index",
    ),
    "_choose_segment_for_line": (
        "y2karaoke.core.components.whisper.whisper_mapping_helpers",
        "_choose_segment_for_line",
    ),
    "_collect_unused_words_in_window": (
        "y2karaoke.core.components.whisper.whisper_mapping_helpers",
        "_collect_unused_words_in_window",
    ),
    "_collect_unused_words_near_line": (
        "y2karaoke.core.components.whisper.whisper_mapping_helpers",
        "_collect_unused_words_near_line",
    ),
    "_compute_gap_window": (
        "y2karaoke.core.components.whisper.whisper_mapping_pipeline",
        "_compute_gap_window",
    ),
    "_dedupe_whisper_segments": (
        "y2karaoke.core.components.whisper.whisper_mapping_helpers",
        "_dedupe_whisper_segments",
    ),
    "_dedupe_whisper_words": (
        "y2karaoke.core.components.whisper.whisper_mapping_helpers",
        "_dedupe_whisper_words",
    ),
    "_enforce_monotonic_line_starts_whisper": (
        "y2karaoke.core.components.whisper.whisper_mapping_post",
        "_enforce_monotonic_line_starts_whisper",
    ),
    "_extend_line_to_trailing_whisper_matches": (
        "y2karaoke.core.components.whisper.whisper_mapping_post",
        "_extend_line_to_trailing_whisper_matches",
    ),
    "_fill_unmatched_gaps": (
        "y2karaoke.core.components.whisper.whisper_mapping_pipeline",
        "_fill_unmatched_gaps",
    ),
    "_filter_and_order_candidates": (
        "y2karaoke.core.components.whisper.whisper_mapping_pipeline",
        "_filter_and_order_candidates",
    ),
    "_find_nearest_word_in_segment": (
        "y2karaoke.core.components.whisper.whisper_mapping_helpers",
        "_find_nearest_word_in_segment",
    ),
    "_find_segment_for_time": (
        "y2karaoke.core.components.whisper.whisper_mapping_helpers",
        "_find_segment_for_time",
    ),
    "_map_lrc_words_to_whisper": (
        "y2karaoke.core.components.whisper.whisper_mapping_pipeline",
        "_map_lrc_words_to_whisper",
    ),
    "_match_assigned_words": (
        "y2karaoke.core.components.whisper.whisper_mapping_pipeline",
        "_match_assigned_words",
    ),
    "_normalize_line_text": (
        "y2karaoke.core.components.whisper.whisper_mapping_helpers",
        "_normalize_line_text",
    ),
    "_prepare_line_context": (
        "y2karaoke.core.components.whisper.whisper_mapping_pipeline",
        "_prepare_line_context",
    ),
    "_pull_late_lines_to_matching_segments": (
        "y2karaoke.core.components.whisper.whisper_mapping_post",
        "_pull_late_lines_to_matching_segments",
    ),
    "_register_word_match": (
        "y2karaoke.core.components.whisper.whisper_mapping_pipeline",
        "_register_word_match",
    ),
    "_resolve_line_overlaps": (
        "y2karaoke.core.components.whisper.whisper_mapping_post",
        "_resolve_line_overlaps",
    ),
    "_retime_short_interjection_lines": (
        "y2karaoke.core.components.whisper.whisper_mapping_post",
        "_retime_short_interjection_lines",
    ),
    "_segment_word_indices": (
        "y2karaoke.core.components.whisper.whisper_mapping_helpers",
        "_segment_word_indices",
    ),
    "_select_best_candidate": (
        "y2karaoke.core.components.whisper.whisper_mapping_pipeline",
        "_select_best_candidate",
    ),
    "_shift_repeated_lines_to_next_whisper": (
        "y2karaoke.core.components.whisper.whisper_mapping_post",
        "_shift_repeated_lines_to_next_whisper",
    ),
    "_snap_first_word_to_whisper_onset": (
        "y2karaoke.core.components.whisper.whisper_mapping_post",
        "_snap_first_word_to_whisper_onset",
    ),
    "_trim_whisper_transcription_by_lyrics": (
        "y2karaoke.core.components.whisper.whisper_mapping_helpers",
        "_trim_whisper_transcription_by_lyrics",
    ),
    "_word_match_score": (
        "y2karaoke.core.components.whisper.whisper_mapping_helpers",
        "_word_match_score",
    ),
}


def __getattr__(name: str):
    if name in _LEGACY_EXPORTS:
        module_name, attr_name = _LEGACY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LEGACY_EXPORTS.keys()))
