"""Alias and re-export wiring for whisper_integration."""

from typing import Any, Dict, List

from . import phonetic_utils
from . import whisper_alignment
from . import whisper_blocks
from . import whisper_cache
from . import whisper_dtw
from . import whisper_mapping
from . import whisper_phonetic_dtw
from . import whisper_utils

ALIAS_EXPORTS: List[str] = [
    "align_lyrics_to_transcription",
    "align_words_to_whisper",
    "align_dtw_whisper",
    "align_hybrid_lrc_whisper",
    "_get_whisper_cache_path",
    "_find_best_cached_whisper_model",
    "_load_whisper_cache",
    "_save_whisper_cache",
    "_model_index",
    "_MODEL_ORDER",
    "_find_best_whisper_match",
    "_extract_lrc_words",
    "_extract_lrc_words_base",
    "_compute_phonetic_costs",
    "_compute_phonetic_costs_base",
    "_extract_alignments_from_path",
    "_extract_alignments_from_path_base",
    "_apply_dtw_alignments",
    "_apply_dtw_alignments_base",
    "align_dtw_whisper_base",
    "_align_dtw_whisper_with_data",
    "_compute_dtw_alignment_metrics",
    "_retime_lines_from_dtw_alignments",
    "_merge_lines_to_whisper_segments",
    "_retime_adjacent_lines_to_whisper_window",
    "_retime_adjacent_lines_to_segment_window",
    "_pull_next_line_into_segment_window",
    "_pull_next_line_into_same_segment",
    "_merge_short_following_line_into_segment",
    "_pull_lines_near_segment_end",
    "_clamp_repeated_line_duration",
    "_tighten_lines_to_whisper_segments",
    "_apply_offset_to_line",
    "_calculate_drift_correction",
    "_fix_ordering_violations",
    "_find_best_whisper_segment",
    "_assess_lrc_quality",
    "_pull_lines_to_best_segments",
    "_fill_vocal_activity_gaps",
    "_pull_lines_forward_for_continuous_vocals",
    "_merge_first_two_lines_if_segment_matches",
    "_whisper_lang_to_epitran",
    "_get_ipa",
    "_phonetic_similarity",
]


def build_aliases() -> Dict[str, Any]:
    """Build the legacy alias dictionary consumed by whisper_integration."""
    aliases: Dict[str, Any] = {}

    aliases["_dedupe_whisper_words"] = whisper_mapping._dedupe_whisper_words
    aliases["_dedupe_whisper_segments"] = whisper_mapping._dedupe_whisper_segments
    aliases["_build_word_to_segment_index"] = (
        whisper_mapping._build_word_to_segment_index
    )
    aliases["_find_segment_for_time"] = whisper_mapping._find_segment_for_time
    aliases["_word_match_score"] = whisper_mapping._word_match_score
    aliases["_find_nearest_word_in_segment"] = (
        whisper_mapping._find_nearest_word_in_segment
    )
    aliases["_normalize_line_text"] = whisper_mapping._normalize_line_text
    aliases["_trim_whisper_transcription_by_lyrics"] = (
        whisper_mapping._trim_whisper_transcription_by_lyrics
    )
    aliases["_choose_segment_for_line"] = whisper_mapping._choose_segment_for_line
    aliases["_segment_word_indices"] = whisper_mapping._segment_word_indices
    aliases["_collect_unused_words_near_line"] = (
        whisper_mapping._collect_unused_words_near_line
    )
    aliases["_collect_unused_words_in_window"] = (
        whisper_mapping._collect_unused_words_in_window
    )
    aliases["_register_word_match"] = whisper_mapping._register_word_match
    aliases["_select_best_candidate"] = whisper_mapping._select_best_candidate
    aliases["_filter_and_order_candidates"] = (
        whisper_mapping._filter_and_order_candidates
    )
    aliases["_prepare_line_context"] = whisper_mapping._prepare_line_context
    aliases["_match_assigned_words"] = whisper_mapping._match_assigned_words
    aliases["_fill_unmatched_gaps"] = whisper_mapping._fill_unmatched_gaps
    aliases["_compute_gap_window"] = whisper_mapping._compute_gap_window
    aliases["_assemble_mapped_line"] = whisper_mapping._assemble_mapped_line
    aliases["_map_lrc_words_to_whisper"] = whisper_mapping._map_lrc_words_to_whisper
    aliases["_build_word_assignments_from_phoneme_path"] = (
        whisper_mapping._build_word_assignments_from_phoneme_path
    )
    aliases["_shift_repeated_lines_to_next_whisper"] = (
        whisper_mapping._shift_repeated_lines_to_next_whisper
    )
    aliases["_enforce_monotonic_line_starts_whisper"] = (
        whisper_mapping._enforce_monotonic_line_starts_whisper
    )
    aliases["_resolve_line_overlaps"] = whisper_mapping._resolve_line_overlaps

    aliases["_redistribute_word_timings_to_line"] = (
        whisper_utils._redistribute_word_timings_to_line
    )
    aliases["_clamp_word_gaps"] = whisper_utils._clamp_word_gaps
    aliases["_cap_word_durations"] = whisper_utils._cap_word_durations

    aliases["_align_dtw_whisper_with_data"] = whisper_dtw._align_dtw_whisper_with_data
    aliases["align_dtw_whisper"] = whisper_dtw.align_dtw_whisper
    aliases["_compute_dtw_alignment_metrics"] = (
        whisper_dtw._compute_dtw_alignment_metrics
    )
    aliases["_retime_lines_from_dtw_alignments"] = (
        whisper_dtw._retime_lines_from_dtw_alignments
    )
    aliases["_extract_alignments_from_path"] = (
        whisper_dtw._extract_alignments_from_path_base
    )
    aliases["_apply_dtw_alignments"] = whisper_dtw._apply_dtw_alignments_base
    aliases["_extract_lrc_words_base"] = whisper_dtw._extract_lrc_words_base
    aliases["_compute_phonetic_costs_base"] = whisper_dtw._compute_phonetic_costs_base
    aliases["_extract_alignments_from_path_base"] = (
        whisper_dtw._extract_alignments_from_path_base
    )
    aliases["_apply_dtw_alignments_base"] = whisper_dtw._apply_dtw_alignments_base
    aliases["align_dtw_whisper_base"] = whisper_dtw.align_dtw_whisper_base

    aliases["_get_whisper_cache_path"] = whisper_cache._get_whisper_cache_path
    aliases["_find_best_cached_whisper_model"] = (
        whisper_cache._find_best_cached_whisper_model
    )
    aliases["_load_whisper_cache"] = whisper_cache._load_whisper_cache
    aliases["_save_whisper_cache"] = whisper_cache._save_whisper_cache
    aliases["_model_index"] = whisper_cache._model_index
    aliases["_MODEL_ORDER"] = whisper_cache._MODEL_ORDER

    aliases["_find_best_whisper_match"] = whisper_phonetic_dtw._find_best_whisper_match
    aliases["align_lyrics_to_transcription"] = (
        whisper_phonetic_dtw.align_lyrics_to_transcription
    )
    aliases["align_words_to_whisper"] = whisper_phonetic_dtw.align_words_to_whisper
    aliases["_assess_lrc_quality"] = whisper_phonetic_dtw._assess_lrc_quality
    aliases["_extract_lrc_words"] = whisper_phonetic_dtw._extract_lrc_words
    aliases["_compute_phonetic_costs"] = whisper_phonetic_dtw._compute_phonetic_costs
    aliases["_compute_phonetic_costs_unbounded"] = (
        whisper_phonetic_dtw._compute_phonetic_costs_unbounded
    )
    aliases["_extract_best_alignment_map"] = (
        whisper_phonetic_dtw._extract_best_alignment_map
    )
    aliases["_extract_lrc_words_all"] = whisper_phonetic_dtw._extract_lrc_words_all
    aliases["_build_dtw_path"] = whisper_phonetic_dtw._build_dtw_path
    aliases["_build_phoneme_dtw_path"] = whisper_phonetic_dtw._build_phoneme_dtw_path
    aliases["_build_syllable_tokens_from_phonemes"] = (
        whisper_phonetic_dtw._build_syllable_tokens_from_phonemes
    )
    aliases["_make_syllable_from_tokens"] = (
        whisper_phonetic_dtw._make_syllable_from_tokens
    )
    aliases["_build_syllable_dtw_path"] = whisper_phonetic_dtw._build_syllable_dtw_path
    aliases["_build_phoneme_tokens_from_lrc_words"] = (
        whisper_phonetic_dtw._build_phoneme_tokens_from_lrc_words
    )
    aliases["_build_phoneme_tokens_from_whisper_words"] = (
        whisper_phonetic_dtw._build_phoneme_tokens_from_whisper_words
    )

    aliases["_assign_lrc_lines_to_blocks"] = whisper_blocks._assign_lrc_lines_to_blocks
    aliases["_text_overlap_score"] = whisper_blocks._text_overlap_score
    aliases["_build_segment_word_info"] = whisper_blocks._build_segment_word_info
    aliases["_assign_lrc_lines_to_segments"] = (
        whisper_blocks._assign_lrc_lines_to_segments
    )
    aliases["_distribute_words_within_segments"] = (
        whisper_blocks._distribute_words_within_segments
    )
    aliases["_build_segment_text_overlap_assignments"] = (
        whisper_blocks._build_segment_text_overlap_assignments
    )
    aliases["_build_block_word_bags"] = whisper_blocks._build_block_word_bags
    aliases["_syl_to_block"] = whisper_blocks._syl_to_block
    aliases["_group_syllables_by_block"] = whisper_blocks._group_syllables_by_block
    aliases["_run_per_block_dtw"] = whisper_blocks._run_per_block_dtw
    aliases["_build_block_segmented_syllable_assignments"] = (
        whisper_blocks._build_block_segmented_syllable_assignments
    )

    aliases["_normalize_word"] = whisper_utils._normalize_word
    aliases["_normalize_words_expanded"] = whisper_utils._normalize_words_expanded
    aliases["_segment_start"] = whisper_utils._segment_start
    aliases["_segment_end"] = whisper_utils._segment_end
    aliases["_get_segment_text"] = whisper_utils._get_segment_text
    aliases["_compute_speech_blocks"] = whisper_utils._compute_speech_blocks
    aliases["_word_idx_to_block"] = whisper_utils._word_idx_to_block
    aliases["_block_time_range"] = whisper_utils._block_time_range
    aliases["_SPEECH_BLOCK_GAP"] = whisper_utils._SPEECH_BLOCK_GAP
    aliases["_build_word_assignments_from_syllable_path"] = (
        whisper_utils._build_word_assignments_from_syllable_path
    )

    aliases["align_hybrid_lrc_whisper"] = whisper_alignment.align_hybrid_lrc_whisper
    aliases["_enforce_monotonic_line_starts"] = (
        whisper_alignment._enforce_monotonic_line_starts
    )
    aliases["_scale_line_to_duration"] = whisper_alignment._scale_line_to_duration
    aliases["_enforce_non_overlapping_lines"] = (
        whisper_alignment._enforce_non_overlapping_lines
    )
    aliases["_merge_lines_to_whisper_segments"] = (
        whisper_alignment._merge_lines_to_whisper_segments
    )
    aliases["_retime_adjacent_lines_to_whisper_window"] = (
        whisper_alignment._retime_adjacent_lines_to_whisper_window
    )
    aliases["_retime_adjacent_lines_to_segment_window"] = (
        whisper_alignment._retime_adjacent_lines_to_segment_window
    )
    aliases["_pull_next_line_into_segment_window"] = (
        whisper_alignment._pull_next_line_into_segment_window
    )
    aliases["_pull_next_line_into_same_segment"] = (
        whisper_alignment._pull_next_line_into_same_segment
    )
    aliases["_merge_short_following_line_into_segment"] = (
        whisper_alignment._merge_short_following_line_into_segment
    )
    aliases["_pull_lines_near_segment_end"] = (
        whisper_alignment._pull_lines_near_segment_end
    )
    aliases["_clamp_repeated_line_duration"] = (
        whisper_alignment._clamp_repeated_line_duration
    )
    aliases["_merge_first_two_lines_if_segment_matches"] = (
        whisper_alignment._merge_first_two_lines_if_segment_matches
    )
    aliases["_tighten_lines_to_whisper_segments"] = (
        whisper_alignment._tighten_lines_to_whisper_segments
    )
    aliases["_pull_lines_to_best_segments"] = (
        whisper_alignment._pull_lines_to_best_segments
    )
    aliases["_drop_duplicate_lines"] = whisper_alignment._drop_duplicate_lines
    aliases["_drop_duplicate_lines_by_timing"] = (
        whisper_alignment._drop_duplicate_lines_by_timing
    )
    aliases["_normalize_line_word_timings"] = (
        whisper_alignment._normalize_line_word_timings
    )
    aliases["_find_best_whisper_segment"] = whisper_alignment._find_best_whisper_segment
    aliases["_apply_offset_to_line"] = whisper_alignment._apply_offset_to_line
    aliases["_calculate_drift_correction"] = (
        whisper_alignment._calculate_drift_correction
    )
    aliases["_interpolate_unmatched_lines"] = (
        whisper_alignment._interpolate_unmatched_lines
    )
    aliases["_refine_unmatched_lines_with_onsets"] = (
        whisper_alignment._refine_unmatched_lines_with_onsets
    )
    aliases["_fix_ordering_violations"] = whisper_alignment._fix_ordering_violations
    aliases["_pull_lines_forward_for_continuous_vocals"] = (
        whisper_alignment._pull_lines_forward_for_continuous_vocals
    )
    aliases["_fill_vocal_activity_gaps"] = whisper_alignment._fill_vocal_activity_gaps

    aliases["_whisper_lang_to_epitran"] = phonetic_utils._whisper_lang_to_epitran
    aliases["_get_ipa"] = phonetic_utils._get_ipa
    aliases["_phonetic_similarity"] = phonetic_utils._phonetic_similarity

    return aliases
