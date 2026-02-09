"""Lyrics timing evaluation against audio analysis.

This module compares lyrics timing (from LRC files) against actual audio
characteristics to evaluate timing quality and identify inconsistencies.
"""

from typing import List, Optional, Tuple, Dict
import numpy as np

from ..utils.logging import get_logger
from .models import Line, Word
from .timing_models import (
    TimingIssue,
    AudioFeatures,
    TimingReport,
    TranscriptionWord,
    TranscriptionSegment,
)
from .phonetic_utils import (
    _VOWEL_REGEX,
    _normalize_text_for_matching,
    _normalize_text_for_phonetic,
    _consonant_skeleton,
    _get_epitran,
    _get_panphon_distance,
    _get_panphon_ft,
    _is_vowel,
    _get_ipa,
    _get_ipa_segs,
    _phonetic_similarity,
    _text_similarity_basic,
    _text_similarity,
    _whisper_lang_to_epitran,
    _epitran_cache,
    _ipa_cache,
    _ipa_segs_cache,
)
from .audio_analysis import (
    extract_audio_features,
    _get_audio_features_cache_path,
    _load_audio_features_cache,
    _save_audio_features_cache,
    _find_silence_regions,
    _compute_silence_overlap,
    _is_time_in_silence,
    _find_vocal_start,
    _find_vocal_end,
    _check_vocal_activity_in_range,
    _check_for_silence_in_range,
)
from .whisper_integration import (
    _get_whisper_cache_path,
    _load_whisper_cache,
    _save_whisper_cache,
    _find_best_cached_whisper_model,
    _find_best_whisper_match,
    _extract_lrc_words,
    _compute_phonetic_costs,
    _extract_alignments_from_path,
    _apply_dtw_alignments,
    _align_dtw_whisper_with_data,
    _compute_dtw_alignment_metrics,
    _retime_lines_from_dtw_alignments,
    _merge_lines_to_whisper_segments,
    _retime_adjacent_lines_to_whisper_window,
    _retime_adjacent_lines_to_segment_window,
    _pull_next_line_into_segment_window,
    _pull_next_line_into_same_segment,
    _merge_short_following_line_into_segment,
    _pull_lines_near_segment_end,
    _clamp_repeated_line_duration,
    _tighten_lines_to_whisper_segments,
    _apply_offset_to_line,
    _calculate_drift_correction,
    _fix_ordering_violations,
    _find_best_whisper_segment,
    _assess_lrc_quality,
    _pull_lines_to_best_segments,
    _model_index,
    _MODEL_ORDER,
    _fill_vocal_activity_gaps,
    _pull_lines_forward_for_continuous_vocals,
    _merge_first_two_lines_if_segment_matches,
    transcribe_vocals,
    align_dtw_whisper,
    align_lyrics_to_transcription,
    align_words_to_whisper,
    align_hybrid_lrc_whisper,
    correct_timing_with_whisper,
    align_lrc_text_to_whisper_timings,
)
from .lrc import parse_lrc_with_timing
from . import timing_evaluator_core
from . import timing_evaluator_comparison
from . import timing_evaluator_correction

logger = get_logger(__name__)

# Re-export evaluation core functions
evaluate_timing = timing_evaluator_core.evaluate_timing
_find_closest_onset = timing_evaluator_core._find_closest_onset
_append_line_spans_silence_issues = (
    timing_evaluator_core._append_line_spans_silence_issues
)
_append_gap_issues = timing_evaluator_core._append_gap_issues
_append_unexpected_pause_issues = timing_evaluator_core._append_unexpected_pause_issues
_check_pause_alignment = timing_evaluator_core._check_pause_alignment
_calculate_pause_score_with_stats = (
    timing_evaluator_core._calculate_pause_score_with_stats
)
_calculate_pause_score = timing_evaluator_core._calculate_pause_score
_generate_summary = timing_evaluator_core._generate_summary

# Re-export comparison functions
compare_sources = timing_evaluator_comparison.compare_sources
select_best_source = timing_evaluator_comparison.select_best_source
print_comparison_report = timing_evaluator_comparison.print_comparison_report

# Re-export correction functions
correct_line_timestamps = timing_evaluator_correction.correct_line_timestamps
fix_spurious_gaps = timing_evaluator_correction.fix_spurious_gaps
_find_best_onset_for_phrase_end = (
    timing_evaluator_correction._find_best_onset_for_phrase_end
)
_find_best_onset_proximity = timing_evaluator_correction._find_best_onset_proximity
_find_best_onset_during_silence = (
    timing_evaluator_correction._find_best_onset_during_silence
)
_find_phrase_end = timing_evaluator_correction._find_phrase_end
_collect_lines_to_merge = timing_evaluator_correction._collect_lines_to_merge
_should_merge_gap = timing_evaluator_correction._should_merge_gap
_merge_lines_with_audio = timing_evaluator_correction._merge_lines_with_audio
