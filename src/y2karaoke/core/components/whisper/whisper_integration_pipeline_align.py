"""Alignment pipeline orchestration for Whisper integration."""

from typing import Any, Callable, Dict, List, Optional, Tuple

from ... import models
from ..alignment import timing_models
from .whisper_integration_align import (
    align_lrc_text_to_whisper_timings_impl as _align_lrc_text_to_whisper_timings_impl,
)
from .whisper_integration_baseline import (
    _clone_lines_for_fallback as _clone_lines_for_fallback_impl,
    _constrain_line_starts_to_baseline as _constrain_line_starts_to_baseline_impl,
    _restore_implausibly_short_lines as _restore_implausibly_short_lines_impl,
    _should_rollback_short_line_degradation as _should_rollback_short_line_degradation_impl,
)
from .whisper_integration_filters import (
    _filter_low_confidence_whisper_words as _filter_low_confidence_whisper_words_impl,
)
from .whisper_integration_hooks import AlignmentPassHooks
from .whisper_integration_retry import (
    retry_improves_alignment as _retry_improves_alignment,
    should_retry_with_aggressive_whisper as _should_retry_with_aggressive_whisper,
)
from .whisper_integration_stages import (
    _run_mapped_line_postpasses as _run_mapped_line_postpasses_impl,
)

_MIN_SEGMENT_OVERLAP_COVERAGE = 0.45


def _build_alignment_pass_kwargs(
    *,
    lines: List[models.Line],
    vocals_path: str,
    language: Optional[str],
    model_size: str,
    temperature: float,
    min_similarity: float,
    audio_features: Optional[timing_models.AudioFeatures],
    lenient_vocal_activity_threshold: float,
    lenient_activity_bonus: float,
    low_word_confidence_threshold: float,
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
) -> Dict[str, Any]:
    hooks = AlignmentPassHooks(
        transcribe_vocals_fn=transcribe_vocals_fn,
        extract_audio_features_fn=extract_audio_features_fn,
        dedupe_whisper_segments_fn=dedupe_whisper_segments_fn,
        trim_whisper_transcription_by_lyrics_fn=trim_whisper_transcription_by_lyrics_fn,
        fill_vocal_activity_gaps_fn=fill_vocal_activity_gaps_fn,
        dedupe_whisper_words_fn=dedupe_whisper_words_fn,
        extract_lrc_words_all_fn=extract_lrc_words_all_fn,
        build_phoneme_tokens_from_lrc_words_fn=build_phoneme_tokens_from_lrc_words_fn,
        build_phoneme_tokens_from_whisper_words_fn=build_phoneme_tokens_from_whisper_words_fn,
        build_syllable_tokens_from_phonemes_fn=build_syllable_tokens_from_phonemes_fn,
        build_segment_text_overlap_assignments_fn=build_segment_text_overlap_assignments_fn,
        build_phoneme_dtw_path_fn=build_phoneme_dtw_path_fn,
        build_word_assignments_from_phoneme_path_fn=build_word_assignments_from_phoneme_path_fn,
        build_block_segmented_syllable_assignments_fn=build_block_segmented_syllable_assignments_fn,
        map_lrc_words_to_whisper_fn=map_lrc_words_to_whisper_fn,
        shift_repeated_lines_to_next_whisper_fn=shift_repeated_lines_to_next_whisper_fn,
        enforce_monotonic_line_starts_whisper_fn=enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=resolve_line_overlaps_fn,
        extend_line_to_trailing_whisper_matches_fn=extend_line_to_trailing_whisper_matches_fn,
        pull_late_lines_to_matching_segments_fn=pull_late_lines_to_matching_segments_fn,
        retime_short_interjection_lines_fn=retime_short_interjection_lines_fn,
        snap_first_word_to_whisper_onset_fn=snap_first_word_to_whisper_onset_fn,
        interpolate_unmatched_lines_fn=interpolate_unmatched_lines_fn,
        refine_unmatched_lines_with_onsets_fn=refine_unmatched_lines_with_onsets_fn,
        pull_lines_forward_for_continuous_vocals_fn=pull_lines_forward_for_continuous_vocals_fn,
    )
    return dict(
        lines=lines,
        vocals_path=vocals_path,
        language=language,
        model_size=model_size,
        temperature=temperature,
        min_similarity=min_similarity,
        audio_features=audio_features,
        lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
        lenient_activity_bonus=lenient_activity_bonus,
        low_word_confidence_threshold=low_word_confidence_threshold,
        hooks=hooks,
        logger=logger,
    )


def _run_alignment_with_optional_aggressive_retry(
    *,
    line_count: int,
    aggressive: bool,
    pass_kwargs: Dict[str, Any],
) -> Tuple[List[models.Line], List[str], Dict[str, float]]:
    mapped_lines, corrections, metrics = _run_alignment_pass(
        aggressive=aggressive,
        **pass_kwargs,
    )
    if not _should_retry_with_aggressive_whisper(
        line_count=line_count, aggressive=aggressive, metrics=metrics
    ):
        return mapped_lines, corrections, metrics

    retry_lines, retry_corrections, retry_metrics = _run_alignment_pass(
        aggressive=True,
        **pass_kwargs,
    )
    if _retry_improves_alignment(metrics, retry_metrics):
        merged_metrics = dict(retry_metrics)
        merged_metrics["aggressive_retry_applied"] = 1.0
        retry_corrections = list(retry_corrections)
        retry_corrections.append(
            "Accepted aggressive Whisper retry after weak initial alignment coverage"
        )
        return retry_lines, retry_corrections, merged_metrics

    retained_metrics = dict(metrics)
    retained_metrics["aggressive_retry_attempted"] = 1.0
    retained_metrics["aggressive_retry_applied"] = 0.0
    return mapped_lines, corrections, retained_metrics


def _run_alignment_pass(
    *,
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
    hooks: AlignmentPassHooks,
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
        transcribe_vocals_fn=hooks.transcribe_vocals_fn,
        extract_audio_features_fn=hooks.extract_audio_features_fn,
        dedupe_whisper_segments_fn=hooks.dedupe_whisper_segments_fn,
        trim_whisper_transcription_by_lyrics_fn=hooks.trim_whisper_transcription_by_lyrics_fn,
        fill_vocal_activity_gaps_fn=hooks.fill_vocal_activity_gaps_fn,
        extract_lrc_words_all_fn=hooks.extract_lrc_words_all_fn,
        build_phoneme_tokens_from_lrc_words_fn=hooks.build_phoneme_tokens_from_lrc_words_fn,
        build_phoneme_tokens_from_whisper_words_fn=hooks.build_phoneme_tokens_from_whisper_words_fn,
        build_syllable_tokens_from_phonemes_fn=hooks.build_syllable_tokens_from_phonemes_fn,
        build_segment_text_overlap_assignments_fn=hooks.build_segment_text_overlap_assignments_fn,
        build_phoneme_dtw_path_fn=hooks.build_phoneme_dtw_path_fn,
        build_word_assignments_from_phoneme_path_fn=hooks.build_word_assignments_from_phoneme_path_fn,
        build_block_segmented_syllable_assignments_fn=hooks.build_block_segmented_syllable_assignments_fn,
        map_lrc_words_to_whisper_fn=hooks.map_lrc_words_to_whisper_fn,
        dedupe_whisper_words_fn=hooks.dedupe_whisper_words_fn,
        interpolate_unmatched_lines_fn=hooks.interpolate_unmatched_lines_fn,
        refine_unmatched_lines_with_onsets_fn=hooks.refine_unmatched_lines_with_onsets_fn,
        shift_repeated_lines_to_next_whisper_fn=hooks.shift_repeated_lines_to_next_whisper_fn,
        extend_line_to_trailing_whisper_matches_fn=hooks.extend_line_to_trailing_whisper_matches_fn,
        pull_late_lines_to_matching_segments_fn=hooks.pull_late_lines_to_matching_segments_fn,
        retime_short_interjection_lines_fn=hooks.retime_short_interjection_lines_fn,
        snap_first_word_to_whisper_onset_fn=hooks.snap_first_word_to_whisper_onset_fn,
        pull_lines_forward_for_continuous_vocals_fn=hooks.pull_lines_forward_for_continuous_vocals_fn,
        enforce_monotonic_line_starts_whisper_fn=hooks.enforce_monotonic_line_starts_whisper_fn,
        resolve_line_overlaps_fn=hooks.resolve_line_overlaps_fn,
        run_mapped_line_postpasses_fn=_run_mapped_line_postpasses_impl,
        constrain_line_starts_to_baseline_fn=_constrain_line_starts_to_baseline_impl,
        should_rollback_short_line_degradation_fn=_should_rollback_short_line_degradation_impl,
        restore_implausibly_short_lines_fn=_restore_implausibly_short_lines_impl,
        clone_lines_for_fallback_fn=_clone_lines_for_fallback_impl,
        filter_low_confidence_whisper_words_fn=_filter_low_confidence_whisper_words_impl,
        min_segment_overlap_coverage=_MIN_SEGMENT_OVERLAP_COVERAGE,
        logger=logger,
    )


def align_lrc_text_to_whisper_timings_impl(
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
    pass_kwargs = _build_alignment_pass_kwargs(
        lines=lines,
        vocals_path=vocals_path,
        language=language,
        model_size=model_size,
        temperature=temperature,
        min_similarity=min_similarity,
        audio_features=audio_features,
        lenient_vocal_activity_threshold=lenient_vocal_activity_threshold,
        lenient_activity_bonus=lenient_activity_bonus,
        low_word_confidence_threshold=low_word_confidence_threshold,
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
        logger=logger,
    )
    return _run_alignment_with_optional_aggressive_retry(
        line_count=len(lines),
        aggressive=aggressive,
        pass_kwargs=pass_kwargs,
    )
