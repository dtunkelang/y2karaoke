"""Lyrics timing evaluation against audio analysis.

This module compares lyrics timing (from LRC files) against actual audio
characteristics to evaluate timing quality and identify inconsistencies.
"""

from ..utils.logging import get_logger
from . import timing_evaluator_core
from . import timing_evaluator_comparison
from . import timing_evaluator_correction
from . import timing_models
from . import whisper_cache
from . import whisper_integration
from . import whisper_dtw
from . import phonetic_utils
from . import audio_analysis

logger = get_logger(__name__)

# Re-export data models from timing_models
TimingIssue = timing_models.TimingIssue
AudioFeatures = timing_models.AudioFeatures
TimingReport = timing_models.TimingReport
TranscriptionWord = timing_models.TranscriptionWord
TranscriptionSegment = timing_models.TranscriptionSegment

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

# Re-export whisper cache functions
_get_whisper_cache_path = whisper_cache._get_whisper_cache_path
_load_whisper_cache = whisper_cache._load_whisper_cache
_save_whisper_cache = whisper_cache._save_whisper_cache
_find_best_cached_whisper_model = whisper_cache._find_best_cached_whisper_model

# Re-export audio analysis functions
extract_audio_features = audio_analysis.extract_audio_features
_check_vocal_activity_in_range = audio_analysis._check_vocal_activity_in_range
_check_for_silence_in_range = audio_analysis._check_for_silence_in_range

# Re-export whisper integration functions
transcribe_vocals = whisper_integration.transcribe_vocals
correct_timing_with_whisper = whisper_integration.correct_timing_with_whisper
align_lyrics_to_transcription = whisper_integration.align_lyrics_to_transcription
align_dtw_whisper = whisper_integration.align_dtw_whisper
align_hybrid_lrc_whisper = whisper_integration.align_hybrid_lrc_whisper
align_words_to_whisper = whisper_integration.align_words_to_whisper
_apply_offset_to_line = whisper_integration._apply_offset_to_line
_calculate_drift_correction = whisper_integration._calculate_drift_correction
_fix_ordering_violations = whisper_integration._fix_ordering_violations
_clamp_repeated_line_duration = whisper_integration._clamp_repeated_line_duration
_find_best_whisper_match = whisper_integration._find_best_whisper_match
_extract_alignments_from_path = whisper_integration._extract_alignments_from_path
_compute_phonetic_costs = whisper_integration._compute_phonetic_costs
_assess_lrc_quality = whisper_integration._assess_lrc_quality
_apply_dtw_alignments = whisper_integration._apply_dtw_alignments
_align_dtw_whisper_with_data = whisper_integration._align_dtw_whisper_with_data
_pull_lines_to_best_segments = whisper_integration._pull_lines_to_best_segments
_retime_adjacent_lines_to_whisper_window = (
    whisper_integration._retime_adjacent_lines_to_whisper_window
)
_retime_adjacent_lines_to_segment_window = (
    whisper_integration._retime_adjacent_lines_to_segment_window
)
_pull_next_line_into_same_segment = (
    whisper_integration._pull_next_line_into_same_segment
)
_merge_short_following_line_into_segment = (
    whisper_integration._merge_short_following_line_into_segment
)
_pull_next_line_into_segment_window = (
    whisper_integration._pull_next_line_into_segment_window
)
_pull_lines_near_segment_end = whisper_integration._pull_lines_near_segment_end

# Re-export DTW functions
_extract_lrc_words = whisper_dtw._extract_lrc_words_base
_retime_lines_from_dtw_alignments = (
    whisper_integration._retime_lines_from_dtw_alignments
)

# Re-export phonetic utilities
_text_similarity = phonetic_utils._text_similarity
_whisper_lang_to_epitran = phonetic_utils._whisper_lang_to_epitran
_get_panphon_distance = phonetic_utils._get_panphon_distance
