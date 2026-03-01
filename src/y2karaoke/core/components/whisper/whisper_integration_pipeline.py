"""Pipeline implementations for Whisper integration entry points."""

from typing import Any, Callable, Dict, List, Optional, Tuple

from ... import models
from ..alignment import timing_models
from .whisper_integration_baseline import (
    _clone_lines_for_fallback as _clone_lines_for_fallback_impl,
    _constrain_line_starts_to_baseline as _constrain_line_starts_to_baseline_impl,
    _implausibly_short_multiword_count as _implausibly_short_multiword_count_impl,
    _restore_implausibly_short_lines as _restore_implausibly_short_lines_impl,
    _should_rollback_short_line_degradation as _should_rollback_short_line_degradation_impl,
)
from .whisper_integration_filters import (
    _filter_low_confidence_whisper_words as _filter_low_confidence_whisper_words_impl,
)
from .whisper_integration_finalize import (
    _apply_low_quality_segment_postpasses as _apply_low_quality_segment_postpasses_impl,
    _finalize_whisper_line_set as _finalize_whisper_line_set_impl,
)
from .whisper_integration_stages import (
    _enforce_mapped_line_stage_invariants as _enforce_mapped_line_stage_invariants_impl,
    _run_mapped_line_postpasses as _run_mapped_line_postpasses_impl,
)
from .whisper_integration_align import (
    align_lrc_text_to_whisper_timings_impl as _align_lrc_text_to_whisper_timings_impl,
)
from .whisper_integration_correct import (
    correct_timing_with_whisper_impl as _correct_timing_with_whisper_impl,
)
from .whisper_integration_transcribe import (
    transcribe_vocals_impl as _transcribe_vocals_impl,
)

_MIN_SEGMENT_OVERLAP_COVERAGE = 0.45


def _clone_lines_for_fallback(lines: List[models.Line]) -> List[models.Line]:
    return _clone_lines_for_fallback_impl(lines)


def _implausibly_short_multiword_count(lines: List[models.Line]) -> int:
    return _implausibly_short_multiword_count_impl(lines)


def _should_rollback_short_line_degradation(
    original_lines: List[models.Line],
    aligned_lines: List[models.Line],
) -> Tuple[bool, int, int]:
    return _should_rollback_short_line_degradation_impl(original_lines, aligned_lines)


def _restore_implausibly_short_lines(
    original_lines: List[models.Line],
    aligned_lines: List[models.Line],
) -> Tuple[List[models.Line], int]:
    return _restore_implausibly_short_lines_impl(original_lines, aligned_lines)


def _constrain_line_starts_to_baseline(
    mapped_lines: List[models.Line],
    baseline_lines: List[models.Line],
    *,
    min_gap: float = 0.01,
) -> List[models.Line]:
    return _constrain_line_starts_to_baseline_impl(
        mapped_lines, baseline_lines, min_gap=min_gap
    )


def _filter_low_confidence_whisper_words(
    words: List[timing_models.TranscriptionWord],
    threshold: float,
    *,
    min_keep_ratio: float = 0.6,
    min_keep_words: int = 20,
) -> List[timing_models.TranscriptionWord]:
    return _filter_low_confidence_whisper_words_impl(
        words,
        threshold,
        min_keep_ratio=min_keep_ratio,
        min_keep_words=min_keep_words,
    )


def _enforce_mapped_line_stage_invariants(
    lines_in: List[models.Line],
    all_words: List[timing_models.TranscriptionWord],
    *,
    enforce_monotonic_line_starts_whisper_fn: Callable[..., Any],
    resolve_line_overlaps_fn: Callable[..., Any],
) -> List[models.Line]:
    return _enforce_mapped_line_stage_invariants_impl(
        lines_in,
        all_words,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
    )


def _run_mapped_line_postpasses(
    *,
    mapped_lines: List[models.Line],
    mapped_lines_set: set[int],
    all_words: List[timing_models.TranscriptionWord],
    transcription: List[timing_models.TranscriptionSegment],
    audio_features: Optional[timing_models.AudioFeatures],
    vocals_path: str,
    epitran_lang: str,
    corrections: List[str],
    interpolate_unmatched_lines_fn: Callable[..., Any],
    refine_unmatched_lines_with_onsets_fn: Callable[..., Any],
    shift_repeated_lines_to_next_whisper_fn: Callable[..., Any],
    extend_line_to_trailing_whisper_matches_fn: Callable[..., Any],
    pull_late_lines_to_matching_segments_fn: Callable[..., Any],
    retime_short_interjection_lines_fn: Callable[..., Any],
    snap_first_word_to_whisper_onset_fn: Callable[..., Any],
    pull_lines_forward_for_continuous_vocals_fn: Callable[..., Any],
    enforce_monotonic_line_starts_whisper_fn: Callable[..., Any],
    resolve_line_overlaps_fn: Callable[..., Any],
) -> Tuple[List[models.Line], List[str]]:
    return _run_mapped_line_postpasses_impl(
        mapped_lines=mapped_lines,
        mapped_lines_set=mapped_lines_set,
        all_words=all_words,
        transcription=transcription,
        audio_features=audio_features,
        vocals_path=vocals_path,
        epitran_lang=epitran_lang,
        corrections=corrections,
        interpolate_unmatched_lines_fn=interpolate_unmatched_lines_fn,
        refine_unmatched_lines_with_onsets_fn=refine_unmatched_lines_with_onsets_fn,
        shift_repeated_lines_to_next_whisper_fn=shift_repeated_lines_to_next_whisper_fn,
        extend_line_to_trailing_whisper_matches_fn=extend_line_to_trailing_whisper_matches_fn,
        pull_late_lines_to_matching_segments_fn=pull_late_lines_to_matching_segments_fn,
        retime_short_interjection_lines_fn=retime_short_interjection_lines_fn,
        snap_first_word_to_whisper_onset_fn=snap_first_word_to_whisper_onset_fn,
        pull_lines_forward_for_continuous_vocals_fn=pull_lines_forward_for_continuous_vocals_fn,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
    )


def transcribe_vocals_impl(
    vocals_path: str,
    language: Optional[str],
    model_size: str,
    aggressive: bool,
    temperature: float,
    *,
    get_whisper_cache_path_fn: Callable[..., Optional[str]],
    find_best_cached_whisper_model_fn: Callable[..., Optional[Tuple[str, str]]],
    load_whisper_cache_fn: Callable[..., Optional[Tuple[Any, Any, str]]],
    save_whisper_cache_fn: Callable[..., None],
    load_whisper_model_class_fn: Callable[[], Any],
    logger,
) -> Tuple[
    List[timing_models.TranscriptionSegment],
    List[timing_models.TranscriptionWord],
    str,
    str,
]:
    return _transcribe_vocals_impl(
        vocals_path,
        language,
        model_size,
        aggressive,
        temperature,
        get_whisper_cache_path_fn=get_whisper_cache_path_fn,
        find_best_cached_whisper_model_fn=find_best_cached_whisper_model_fn,
        load_whisper_cache_fn=load_whisper_cache_fn,
        save_whisper_cache_fn=save_whisper_cache_fn,
        load_whisper_model_class_fn=load_whisper_model_class_fn,
        logger=logger,
    )


def align_lrc_text_to_whisper_timings_impl(  # noqa: C901
    lines: List[models.Line],
    vocals_path: str,
    language: Optional[str],
    model_size: str,
    aggressive: bool,
    temperature: float,
    min_similarity: float,
    audio_features: Optional[timing_models.AudioFeatures],
    lenient_vocal_activity_threshold: float,
    lenient_activity_bonus: float,
    low_word_confidence_threshold: float,
    *,
    transcribe_vocals_fn: Callable[..., Tuple[Any, Any, str, str]],
    extract_audio_features_fn: Callable[..., Optional[timing_models.AudioFeatures]],
    dedupe_whisper_segments_fn: Callable[..., Any],
    trim_whisper_transcription_by_lyrics_fn: Callable[..., Any],
    fill_vocal_activity_gaps_fn: Callable[..., Any],
    dedupe_whisper_words_fn: Callable[..., Any],
    extract_lrc_words_all_fn: Callable[..., Any],
    build_phoneme_tokens_from_lrc_words_fn: Callable[..., Any],
    build_phoneme_tokens_from_whisper_words_fn: Callable[..., Any],
    build_syllable_tokens_from_phonemes_fn: Callable[..., Any],
    build_segment_text_overlap_assignments_fn: Callable[..., Any],
    build_phoneme_dtw_path_fn: Callable[..., Any],
    build_word_assignments_from_phoneme_path_fn: Callable[..., Any],
    build_block_segmented_syllable_assignments_fn: Callable[..., Any],
    map_lrc_words_to_whisper_fn: Callable[..., Any],
    shift_repeated_lines_to_next_whisper_fn: Callable[..., Any],
    enforce_monotonic_line_starts_whisper_fn: Callable[..., Any],
    resolve_line_overlaps_fn: Callable[..., Any],
    extend_line_to_trailing_whisper_matches_fn: Callable[..., Any],
    pull_late_lines_to_matching_segments_fn: Callable[..., Any],
    retime_short_interjection_lines_fn: Callable[..., Any],
    snap_first_word_to_whisper_onset_fn: Callable[..., Any],
    interpolate_unmatched_lines_fn: Callable[..., Any],
    refine_unmatched_lines_with_onsets_fn: Callable[..., Any],
    pull_lines_forward_for_continuous_vocals_fn: Callable[..., Any],
    logger,
) -> Tuple[List[models.Line], List[str], Dict[str, float]]:
    return _align_lrc_text_to_whisper_timings_impl(
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
        transcribe_vocals_fn=transcribe_vocals_fn,
        extract_audio_features_fn=extract_audio_features_fn,
        dedupe_whisper_segments_fn=dedupe_whisper_segments_fn,
        trim_whisper_transcription_by_lyrics_fn=trim_whisper_transcription_by_lyrics_fn,
        fill_vocal_activity_gaps_fn=fill_vocal_activity_gaps_fn,
        extract_lrc_words_all_fn=extract_lrc_words_all_fn,
        build_phoneme_tokens_from_lrc_words_fn=build_phoneme_tokens_from_lrc_words_fn,
        build_phoneme_tokens_from_whisper_words_fn=build_phoneme_tokens_from_whisper_words_fn,
        build_syllable_tokens_from_phonemes_fn=build_syllable_tokens_from_phonemes_fn,
        build_segment_text_overlap_assignments_fn=build_segment_text_overlap_assignments_fn,
        build_phoneme_dtw_path_fn=build_phoneme_dtw_path_fn,
        build_word_assignments_from_phoneme_path_fn=build_word_assignments_from_phoneme_path_fn,
        build_block_segmented_syllable_assignments_fn=build_block_segmented_syllable_assignments_fn,
        map_lrc_words_to_whisper_fn=map_lrc_words_to_whisper_fn,
        dedupe_whisper_words_fn=dedupe_whisper_words_fn,
        interpolate_unmatched_lines_fn=interpolate_unmatched_lines_fn,
        refine_unmatched_lines_with_onsets_fn=refine_unmatched_lines_with_onsets_fn,
        shift_repeated_lines_to_next_whisper_fn=shift_repeated_lines_to_next_whisper_fn,
        extend_line_to_trailing_whisper_matches_fn=extend_line_to_trailing_whisper_matches_fn,
        pull_late_lines_to_matching_segments_fn=pull_late_lines_to_matching_segments_fn,
        retime_short_interjection_lines_fn=retime_short_interjection_lines_fn,
        snap_first_word_to_whisper_onset_fn=snap_first_word_to_whisper_onset_fn,
        pull_lines_forward_for_continuous_vocals_fn=pull_lines_forward_for_continuous_vocals_fn,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
        run_mapped_line_postpasses_fn=_run_mapped_line_postpasses,
        constrain_line_starts_to_baseline_fn=_constrain_line_starts_to_baseline,
        should_rollback_short_line_degradation_fn=_should_rollback_short_line_degradation,
        restore_implausibly_short_lines_fn=_restore_implausibly_short_lines,
        clone_lines_for_fallback_fn=_clone_lines_for_fallback,
        filter_low_confidence_whisper_words_fn=_filter_low_confidence_whisper_words,
        min_segment_overlap_coverage=_MIN_SEGMENT_OVERLAP_COVERAGE,
        logger=logger,
    )


def correct_timing_with_whisper_impl(  # noqa: C901
    lines: List[models.Line],
    vocals_path: str,
    language: Optional[str],
    model_size: str,
    aggressive: bool,
    temperature: float,
    trust_lrc_threshold: float,
    correct_lrc_threshold: float,
    force_dtw: bool,
    audio_features: Optional[timing_models.AudioFeatures],
    lenient_vocal_activity_threshold: float,
    lenient_activity_bonus: float,
    low_word_confidence_threshold: float,
    *,
    transcribe_vocals_fn: Callable[..., Tuple[Any, Any, str, str]],
    extract_audio_features_fn: Callable[..., Optional[timing_models.AudioFeatures]],
    trim_whisper_transcription_by_lyrics_fn: Callable[..., Any],
    fill_vocal_activity_gaps_fn: Callable[..., Any],
    assess_lrc_quality_fn: Callable[..., Any],
    align_hybrid_lrc_whisper_fn: Callable[..., Any],
    align_dtw_whisper_with_data_fn: Callable[..., Any],
    retime_lines_from_dtw_alignments_fn: Callable[..., Any],
    merge_first_two_lines_if_segment_matches_fn: Callable[..., Any],
    retime_adjacent_lines_to_whisper_window_fn: Callable[..., Any],
    retime_adjacent_lines_to_segment_window_fn: Callable[..., Any],
    pull_next_line_into_segment_window_fn: Callable[..., Any],
    pull_lines_near_segment_end_fn: Callable[..., Any],
    pull_next_line_into_same_segment_fn: Callable[..., Any],
    merge_lines_to_whisper_segments_fn: Callable[..., Any],
    tighten_lines_to_whisper_segments_fn: Callable[..., Any],
    pull_lines_to_best_segments_fn: Callable[..., Any],
    fix_ordering_violations_fn: Callable[..., Any],
    normalize_line_word_timings_fn: Callable[..., Any],
    enforce_monotonic_line_starts_fn: Callable[..., Any],
    enforce_non_overlapping_lines_fn: Callable[..., Any],
    merge_short_following_line_into_segment_fn: Callable[..., Any],
    clamp_repeated_line_duration_fn: Callable[..., Any],
    drop_duplicate_lines_fn: Callable[..., Any],
    drop_duplicate_lines_by_timing_fn: Callable[..., Any],
    pull_lines_forward_for_continuous_vocals_fn: Callable[..., Any],
    logger,
) -> Tuple[List[models.Line], List[str], Dict[str, float]]:
    return _correct_timing_with_whisper_impl(
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
        transcribe_vocals_fn=transcribe_vocals_fn,
        extract_audio_features_fn=extract_audio_features_fn,
        trim_whisper_transcription_by_lyrics_fn=trim_whisper_transcription_by_lyrics_fn,
        fill_vocal_activity_gaps_fn=fill_vocal_activity_gaps_fn,
        assess_lrc_quality_fn=assess_lrc_quality_fn,
        align_hybrid_lrc_whisper_fn=align_hybrid_lrc_whisper_fn,
        align_dtw_whisper_with_data_fn=align_dtw_whisper_with_data_fn,
        retime_lines_from_dtw_alignments_fn=retime_lines_from_dtw_alignments_fn,
        apply_low_quality_segment_postpasses_fn=_apply_low_quality_segment_postpasses,
        finalize_whisper_line_set_fn=_finalize_whisper_line_set,
        constrain_line_starts_to_baseline_fn=_constrain_line_starts_to_baseline,
        should_rollback_short_line_degradation_fn=_should_rollback_short_line_degradation,
        restore_implausibly_short_lines_fn=_restore_implausibly_short_lines,
        clone_lines_for_fallback_fn=_clone_lines_for_fallback,
        logger=logger,
        merge_first_two_lines_if_segment_matches_fn=merge_first_two_lines_if_segment_matches_fn,
        retime_adjacent_lines_to_whisper_window_fn=retime_adjacent_lines_to_whisper_window_fn,
        retime_adjacent_lines_to_segment_window_fn=retime_adjacent_lines_to_segment_window_fn,
        pull_next_line_into_segment_window_fn=pull_next_line_into_segment_window_fn,
        pull_lines_near_segment_end_fn=pull_lines_near_segment_end_fn,
        pull_next_line_into_same_segment_fn=pull_next_line_into_same_segment_fn,
        merge_lines_to_whisper_segments_fn=merge_lines_to_whisper_segments_fn,
        tighten_lines_to_whisper_segments_fn=tighten_lines_to_whisper_segments_fn,
        pull_lines_to_best_segments_fn=pull_lines_to_best_segments_fn,
        fix_ordering_violations_fn=fix_ordering_violations_fn,
        normalize_line_word_timings_fn=normalize_line_word_timings_fn,
        enforce_monotonic_line_starts_fn=enforce_monotonic_line_starts_fn,
        enforce_non_overlapping_lines_fn=enforce_non_overlapping_lines_fn,
        merge_short_following_line_into_segment_fn=merge_short_following_line_into_segment_fn,
        clamp_repeated_line_duration_fn=clamp_repeated_line_duration_fn,
        drop_duplicate_lines_fn=drop_duplicate_lines_fn,
        drop_duplicate_lines_by_timing_fn=drop_duplicate_lines_by_timing_fn,
        pull_lines_forward_for_continuous_vocals_fn=pull_lines_forward_for_continuous_vocals_fn,
    )


def _apply_low_quality_segment_postpasses(
    *,
    aligned_lines: List[models.Line],
    alignments: List[str],
    transcription: List[timing_models.TranscriptionSegment],
    epitran_lang: str,
    merge_first_two_lines_if_segment_matches_fn: Callable[..., Any],
    retime_adjacent_lines_to_whisper_window_fn: Callable[..., Any],
    retime_adjacent_lines_to_segment_window_fn: Callable[..., Any],
    pull_next_line_into_segment_window_fn: Callable[..., Any],
    pull_lines_near_segment_end_fn: Callable[..., Any],
    pull_next_line_into_same_segment_fn: Callable[..., Any],
    merge_lines_to_whisper_segments_fn: Callable[..., Any],
    tighten_lines_to_whisper_segments_fn: Callable[..., Any],
    pull_lines_to_best_segments_fn: Callable[..., Any],
) -> Tuple[List[models.Line], List[str]]:
    return _apply_low_quality_segment_postpasses_impl(
        aligned_lines=aligned_lines,
        alignments=alignments,
        transcription=transcription,
        epitran_lang=epitran_lang,
        merge_first_two_lines_if_segment_matches_fn=merge_first_two_lines_if_segment_matches_fn,
        retime_adjacent_lines_to_whisper_window_fn=retime_adjacent_lines_to_whisper_window_fn,
        retime_adjacent_lines_to_segment_window_fn=retime_adjacent_lines_to_segment_window_fn,
        pull_next_line_into_segment_window_fn=pull_next_line_into_segment_window_fn,
        pull_lines_near_segment_end_fn=pull_lines_near_segment_end_fn,
        pull_next_line_into_same_segment_fn=pull_next_line_into_same_segment_fn,
        merge_lines_to_whisper_segments_fn=merge_lines_to_whisper_segments_fn,
        tighten_lines_to_whisper_segments_fn=tighten_lines_to_whisper_segments_fn,
        pull_lines_to_best_segments_fn=pull_lines_to_best_segments_fn,
    )


def _finalize_whisper_line_set(
    *,
    source_lines: List[models.Line],
    aligned_lines: List[models.Line],
    alignments: List[str],
    transcription: List[timing_models.TranscriptionSegment],
    epitran_lang: str,
    force_dtw: bool,
    audio_features: Optional[timing_models.AudioFeatures],
    fix_ordering_violations_fn: Callable[..., Any],
    normalize_line_word_timings_fn: Callable[..., Any],
    enforce_monotonic_line_starts_fn: Callable[..., Any],
    enforce_non_overlapping_lines_fn: Callable[..., Any],
    pull_lines_near_segment_end_fn: Callable[..., Any],
    merge_short_following_line_into_segment_fn: Callable[..., Any],
    clamp_repeated_line_duration_fn: Callable[..., Any],
    drop_duplicate_lines_fn: Callable[..., Any],
    drop_duplicate_lines_by_timing_fn: Callable[..., Any],
    pull_lines_forward_for_continuous_vocals_fn: Callable[..., Any],
) -> Tuple[List[models.Line], List[str]]:
    return _finalize_whisper_line_set_impl(
        source_lines=source_lines,
        aligned_lines=aligned_lines,
        alignments=alignments,
        transcription=transcription,
        epitran_lang=epitran_lang,
        force_dtw=force_dtw,
        audio_features=audio_features,
        fix_ordering_violations_fn=fix_ordering_violations_fn,
        normalize_line_word_timings_fn=normalize_line_word_timings_fn,
        enforce_monotonic_line_starts_fn=enforce_monotonic_line_starts_fn,
        enforce_non_overlapping_lines_fn=enforce_non_overlapping_lines_fn,
        pull_lines_near_segment_end_fn=pull_lines_near_segment_end_fn,
        merge_short_following_line_into_segment_fn=merge_short_following_line_into_segment_fn,
        clamp_repeated_line_duration_fn=clamp_repeated_line_duration_fn,
        drop_duplicate_lines_fn=drop_duplicate_lines_fn,
        drop_duplicate_lines_by_timing_fn=drop_duplicate_lines_by_timing_fn,
        pull_lines_forward_for_continuous_vocals_fn=pull_lines_forward_for_continuous_vocals_fn,
    )
