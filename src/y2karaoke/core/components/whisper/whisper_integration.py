"""Whisper-based transcription and alignment for lyrics."""

from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

from ....utils.logging import get_logger
from ... import models
from ...audio_analysis import extract_audio_features
from ..alignment import timing_models
from .whisper_integration_aliases import ALIAS_EXPORTS, build_aliases
from .whisper_integration_pipeline import (
    align_lrc_text_to_whisper_timings_impl,
    correct_timing_with_whisper_impl,
    transcribe_vocals_impl,
)

logger = get_logger(__name__)

_ALIASES = build_aliases()
globals().update(_ALIASES)

_get_whisper_cache_path = _ALIASES["_get_whisper_cache_path"]
_find_best_cached_whisper_model = _ALIASES["_find_best_cached_whisper_model"]
_load_whisper_cache = _ALIASES["_load_whisper_cache"]
_save_whisper_cache = _ALIASES["_save_whisper_cache"]
align_lyrics_to_transcription = _ALIASES["align_lyrics_to_transcription"]
align_dtw_whisper = _ALIASES["align_dtw_whisper"]
align_words_to_whisper = _ALIASES["align_words_to_whisper"]
_dedupe_whisper_segments = _ALIASES["_dedupe_whisper_segments"]
_trim_whisper_transcription_by_lyrics = _ALIASES[
    "_trim_whisper_transcription_by_lyrics"
]
_fill_vocal_activity_gaps = _ALIASES["_fill_vocal_activity_gaps"]
_dedupe_whisper_words = _ALIASES["_dedupe_whisper_words"]
_extract_lrc_words_all = _ALIASES["_extract_lrc_words_all"]
_build_phoneme_tokens_from_lrc_words = _ALIASES["_build_phoneme_tokens_from_lrc_words"]
_build_phoneme_tokens_from_whisper_words = _ALIASES[
    "_build_phoneme_tokens_from_whisper_words"
]
_build_syllable_tokens_from_phonemes = _ALIASES["_build_syllable_tokens_from_phonemes"]
_build_segment_text_overlap_assignments = _ALIASES[
    "_build_segment_text_overlap_assignments"
]
_build_phoneme_dtw_path = _ALIASES["_build_phoneme_dtw_path"]
_build_word_assignments_from_phoneme_path = _ALIASES[
    "_build_word_assignments_from_phoneme_path"
]
_build_block_segmented_syllable_assignments = _ALIASES[
    "_build_block_segmented_syllable_assignments"
]
_map_lrc_words_to_whisper = _ALIASES["_map_lrc_words_to_whisper"]
_shift_repeated_lines_to_next_whisper = _ALIASES[
    "_shift_repeated_lines_to_next_whisper"
]
_enforce_monotonic_line_starts_whisper = _ALIASES[
    "_enforce_monotonic_line_starts_whisper"
]
_resolve_line_overlaps = _ALIASES["_resolve_line_overlaps"]
_extend_line_to_trailing_whisper_matches = _ALIASES[
    "_extend_line_to_trailing_whisper_matches"
]
_pull_late_lines_to_matching_segments = _ALIASES[
    "_pull_late_lines_to_matching_segments"
]
_retime_short_interjection_lines = _ALIASES["_retime_short_interjection_lines"]
_snap_first_word_to_whisper_onset = _ALIASES["_snap_first_word_to_whisper_onset"]
_interpolate_unmatched_lines = _ALIASES["_interpolate_unmatched_lines"]
_refine_unmatched_lines_with_onsets = _ALIASES["_refine_unmatched_lines_with_onsets"]
_assess_lrc_quality = _ALIASES["_assess_lrc_quality"]
align_hybrid_lrc_whisper = _ALIASES["align_hybrid_lrc_whisper"]
_align_dtw_whisper_with_data = _ALIASES["_align_dtw_whisper_with_data"]
_extract_alignments_from_path = _ALIASES["_extract_alignments_from_path"]
_compute_phonetic_costs = _ALIASES["_compute_phonetic_costs"]
_apply_dtw_alignments = _ALIASES["_apply_dtw_alignments"]
_apply_offset_to_line = _ALIASES["_apply_offset_to_line"]
_calculate_drift_correction = _ALIASES["_calculate_drift_correction"]
_find_best_whisper_match = _ALIASES["_find_best_whisper_match"]
_retime_lines_from_dtw_alignments = _ALIASES["_retime_lines_from_dtw_alignments"]
_merge_first_two_lines_if_segment_matches = _ALIASES[
    "_merge_first_two_lines_if_segment_matches"
]
_retime_adjacent_lines_to_whisper_window = _ALIASES[
    "_retime_adjacent_lines_to_whisper_window"
]
_retime_adjacent_lines_to_segment_window = _ALIASES[
    "_retime_adjacent_lines_to_segment_window"
]
_pull_next_line_into_segment_window = _ALIASES["_pull_next_line_into_segment_window"]
_pull_lines_near_segment_end = _ALIASES["_pull_lines_near_segment_end"]
_pull_next_line_into_same_segment = _ALIASES["_pull_next_line_into_same_segment"]
_merge_lines_to_whisper_segments = _ALIASES["_merge_lines_to_whisper_segments"]
_tighten_lines_to_whisper_segments = _ALIASES["_tighten_lines_to_whisper_segments"]
_pull_lines_to_best_segments = _ALIASES["_pull_lines_to_best_segments"]
_fix_ordering_violations = _ALIASES["_fix_ordering_violations"]
_normalize_line_word_timings = _ALIASES["_normalize_line_word_timings"]
_enforce_monotonic_line_starts = _ALIASES["_enforce_monotonic_line_starts"]
_enforce_non_overlapping_lines = _ALIASES["_enforce_non_overlapping_lines"]
_merge_short_following_line_into_segment = _ALIASES[
    "_merge_short_following_line_into_segment"
]
_clamp_repeated_line_duration = _ALIASES["_clamp_repeated_line_duration"]
_drop_duplicate_lines = _ALIASES["_drop_duplicate_lines"]
_drop_duplicate_lines_by_timing = _ALIASES["_drop_duplicate_lines_by_timing"]
_pull_lines_forward_for_continuous_vocals = _ALIASES[
    "_pull_lines_forward_for_continuous_vocals"
]
_get_ipa = _ALIASES["_get_ipa"]

__all__ = [
    "transcribe_vocals",
    "correct_timing_with_whisper",
    "align_lrc_text_to_whisper_timings",
] + ALIAS_EXPORTS


@contextmanager
def use_whisper_integration_hooks(
    *,
    transcribe_vocals_fn=None,
    extract_audio_features_fn=None,
    assess_lrc_quality_fn=None,
    align_hybrid_lrc_whisper_fn=None,
    align_dtw_whisper_with_data_fn=None,
    load_whisper_model_class_fn=None,
    get_whisper_cache_path_fn=None,
    load_whisper_cache_fn=None,
    save_whisper_cache_fn=None,
    retime_lines_from_dtw_alignments_fn=None,
    get_ipa_fn=None,
):
    """Temporarily override integration collaborators for tests."""
    global transcribe_vocals, extract_audio_features
    global _assess_lrc_quality, align_hybrid_lrc_whisper
    global _align_dtw_whisper_with_data
    global _load_whisper_model_class
    global _get_whisper_cache_path, _load_whisper_cache, _save_whisper_cache
    global _retime_lines_from_dtw_alignments
    global _get_ipa

    prev_transcribe_vocals = transcribe_vocals
    prev_extract_audio_features = extract_audio_features
    prev_assess_lrc_quality = _assess_lrc_quality
    prev_align_hybrid = align_hybrid_lrc_whisper
    prev_align_dtw_data = _align_dtw_whisper_with_data
    prev_load_model = _load_whisper_model_class
    prev_get_cache_path = _get_whisper_cache_path
    prev_load_cache = _load_whisper_cache
    prev_save_cache = _save_whisper_cache
    prev_retime_from_dtw = _retime_lines_from_dtw_alignments
    prev_get_ipa = _get_ipa

    if transcribe_vocals_fn is not None:
        transcribe_vocals = transcribe_vocals_fn
    if extract_audio_features_fn is not None:
        extract_audio_features = extract_audio_features_fn
    if assess_lrc_quality_fn is not None:
        _assess_lrc_quality = assess_lrc_quality_fn
    if align_hybrid_lrc_whisper_fn is not None:
        align_hybrid_lrc_whisper = align_hybrid_lrc_whisper_fn
    if align_dtw_whisper_with_data_fn is not None:
        _align_dtw_whisper_with_data = align_dtw_whisper_with_data_fn
    if load_whisper_model_class_fn is not None:
        _load_whisper_model_class = load_whisper_model_class_fn
    if get_whisper_cache_path_fn is not None:
        _get_whisper_cache_path = get_whisper_cache_path_fn
    if load_whisper_cache_fn is not None:
        _load_whisper_cache = load_whisper_cache_fn
    if save_whisper_cache_fn is not None:
        _save_whisper_cache = save_whisper_cache_fn
    if retime_lines_from_dtw_alignments_fn is not None:
        _retime_lines_from_dtw_alignments = retime_lines_from_dtw_alignments_fn
    if get_ipa_fn is not None:
        _get_ipa = get_ipa_fn

    try:
        yield
    finally:
        transcribe_vocals = prev_transcribe_vocals
        extract_audio_features = prev_extract_audio_features
        _assess_lrc_quality = prev_assess_lrc_quality
        align_hybrid_lrc_whisper = prev_align_hybrid
        _align_dtw_whisper_with_data = prev_align_dtw_data
        _load_whisper_model_class = prev_load_model
        _get_whisper_cache_path = prev_get_cache_path
        _load_whisper_cache = prev_load_cache
        _save_whisper_cache = prev_save_cache
        _retime_lines_from_dtw_alignments = prev_retime_from_dtw
        _get_ipa = prev_get_ipa


def _load_whisper_model_class():
    from faster_whisper import WhisperModel  # type: ignore

    return WhisperModel


def transcribe_vocals(
    vocals_path: str,
    language: Optional[str] = None,
    model_size: str = "large",
    aggressive: bool = False,
    temperature: float = 0.0,
) -> Tuple[
    List[timing_models.TranscriptionSegment],
    List[timing_models.TranscriptionWord],
    str,
    str,
]:
    """Transcribe vocals using Whisper and cache results."""
    return transcribe_vocals_impl(
        vocals_path,
        language,
        model_size,
        aggressive,
        temperature,
        get_whisper_cache_path_fn=_get_whisper_cache_path,
        find_best_cached_whisper_model_fn=_find_best_cached_whisper_model,
        load_whisper_cache_fn=_load_whisper_cache,
        save_whisper_cache_fn=_save_whisper_cache,
        load_whisper_model_class_fn=_load_whisper_model_class,
        logger=logger,
    )


_TIME_DRIFT_THRESHOLD = 0.8


def align_lrc_text_to_whisper_timings(
    lines: List[models.Line],
    vocals_path: str,
    language: Optional[str] = None,
    model_size: str = "large",
    aggressive: bool = False,
    temperature: float = 0.0,
    min_similarity: float = 0.15,
    audio_features: Optional[timing_models.AudioFeatures] = None,
    lenient_vocal_activity_threshold: float = 0.3,
    lenient_activity_bonus: float = 0.4,
    low_word_confidence_threshold: float = 0.5,
) -> Tuple[List[models.Line], List[str], Dict[str, float]]:
    """Align LRC text to Whisper timings using phonetic DTW."""
    return align_lrc_text_to_whisper_timings_impl(
        lines,
        vocals_path,
        language,
        model_size,
        aggressive,
        temperature,
        min_similarity,
        audio_features,
        lenient_vocal_activity_threshold,
        lenient_activity_bonus,
        low_word_confidence_threshold,
        transcribe_vocals_fn=transcribe_vocals,
        extract_audio_features_fn=extract_audio_features,
        dedupe_whisper_segments_fn=_dedupe_whisper_segments,
        trim_whisper_transcription_by_lyrics_fn=_trim_whisper_transcription_by_lyrics,
        fill_vocal_activity_gaps_fn=_fill_vocal_activity_gaps,
        dedupe_whisper_words_fn=_dedupe_whisper_words,
        extract_lrc_words_all_fn=_extract_lrc_words_all,
        build_phoneme_tokens_from_lrc_words_fn=_build_phoneme_tokens_from_lrc_words,
        build_phoneme_tokens_from_whisper_words_fn=(
            _build_phoneme_tokens_from_whisper_words
        ),
        build_syllable_tokens_from_phonemes_fn=_build_syllable_tokens_from_phonemes,
        build_segment_text_overlap_assignments_fn=(
            _build_segment_text_overlap_assignments
        ),
        build_phoneme_dtw_path_fn=_build_phoneme_dtw_path,
        build_word_assignments_from_phoneme_path_fn=(
            _build_word_assignments_from_phoneme_path
        ),
        build_block_segmented_syllable_assignments_fn=(
            _build_block_segmented_syllable_assignments
        ),
        map_lrc_words_to_whisper_fn=_map_lrc_words_to_whisper,
        shift_repeated_lines_to_next_whisper_fn=_shift_repeated_lines_to_next_whisper,
        enforce_monotonic_line_starts_whisper_fn=(
            _enforce_monotonic_line_starts_whisper
        ),
        resolve_line_overlaps_fn=_resolve_line_overlaps,
        extend_line_to_trailing_whisper_matches_fn=(
            _extend_line_to_trailing_whisper_matches
        ),
        pull_late_lines_to_matching_segments_fn=(_pull_late_lines_to_matching_segments),
        retime_short_interjection_lines_fn=_retime_short_interjection_lines,
        snap_first_word_to_whisper_onset_fn=_snap_first_word_to_whisper_onset,
        interpolate_unmatched_lines_fn=_interpolate_unmatched_lines,
        refine_unmatched_lines_with_onsets_fn=_refine_unmatched_lines_with_onsets,
        pull_lines_forward_for_continuous_vocals_fn=(
            _pull_lines_forward_for_continuous_vocals
        ),
        logger=logger,
    )


def correct_timing_with_whisper(
    lines: List[models.Line],
    vocals_path: str,
    language: Optional[str] = None,
    model_size: str = "large",
    aggressive: bool = False,
    temperature: float = 0.0,
    trust_lrc_threshold: float = 1.0,
    correct_lrc_threshold: float = 1.5,
    force_dtw: bool = False,
    audio_features: Optional[timing_models.AudioFeatures] = None,
    lenient_vocal_activity_threshold: float = 0.3,
    lenient_activity_bonus: float = 0.4,
    low_word_confidence_threshold: float = 0.5,
) -> Tuple[List[models.Line], List[str], Dict[str, float]]:
    """Correct lyrics timing using Whisper transcription (adaptive approach)."""
    return correct_timing_with_whisper_impl(
        lines,
        vocals_path,
        language,
        model_size,
        aggressive,
        temperature,
        trust_lrc_threshold,
        correct_lrc_threshold,
        force_dtw,
        audio_features,
        lenient_vocal_activity_threshold,
        lenient_activity_bonus,
        low_word_confidence_threshold,
        transcribe_vocals_fn=transcribe_vocals,
        extract_audio_features_fn=extract_audio_features,
        trim_whisper_transcription_by_lyrics_fn=_trim_whisper_transcription_by_lyrics,
        fill_vocal_activity_gaps_fn=_fill_vocal_activity_gaps,
        assess_lrc_quality_fn=_assess_lrc_quality,
        align_hybrid_lrc_whisper_fn=align_hybrid_lrc_whisper,
        align_dtw_whisper_with_data_fn=_align_dtw_whisper_with_data,
        retime_lines_from_dtw_alignments_fn=_retime_lines_from_dtw_alignments,
        merge_first_two_lines_if_segment_matches_fn=(
            _merge_first_two_lines_if_segment_matches
        ),
        retime_adjacent_lines_to_whisper_window_fn=(
            _retime_adjacent_lines_to_whisper_window
        ),
        retime_adjacent_lines_to_segment_window_fn=(
            _retime_adjacent_lines_to_segment_window
        ),
        pull_next_line_into_segment_window_fn=_pull_next_line_into_segment_window,
        pull_lines_near_segment_end_fn=_pull_lines_near_segment_end,
        pull_next_line_into_same_segment_fn=_pull_next_line_into_same_segment,
        merge_lines_to_whisper_segments_fn=_merge_lines_to_whisper_segments,
        tighten_lines_to_whisper_segments_fn=_tighten_lines_to_whisper_segments,
        pull_lines_to_best_segments_fn=_pull_lines_to_best_segments,
        fix_ordering_violations_fn=_fix_ordering_violations,
        normalize_line_word_timings_fn=_normalize_line_word_timings,
        enforce_monotonic_line_starts_fn=_enforce_monotonic_line_starts,
        enforce_non_overlapping_lines_fn=_enforce_non_overlapping_lines,
        merge_short_following_line_into_segment_fn=(
            _merge_short_following_line_into_segment
        ),
        clamp_repeated_line_duration_fn=_clamp_repeated_line_duration,
        drop_duplicate_lines_fn=_drop_duplicate_lines,
        drop_duplicate_lines_by_timing_fn=_drop_duplicate_lines_by_timing,
        pull_lines_forward_for_continuous_vocals_fn=(
            _pull_lines_forward_for_continuous_vocals
        ),
        logger=logger,
    )
