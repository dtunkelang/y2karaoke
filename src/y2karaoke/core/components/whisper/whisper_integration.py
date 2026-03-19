"""Whisper-based transcription and alignment for lyrics."""

from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

from ....utils.logging import get_logger
from ... import models
from ...audio_analysis import extract_audio_features
from ..alignment import timing_models
from .whisper_integration_aliases import build_aliases
from .whisper_integration_pipeline import (
    align_lrc_text_to_whisper_timings_impl,
    correct_timing_with_whisper_impl,
    transcribe_vocals_impl,
)
from .whisper_integration_baseline import (
    _clone_lines_for_fallback as _clone_lines_for_fallback_impl,
    _constrain_line_starts_to_baseline as _constrain_line_starts_to_baseline_impl,
    _restore_implausibly_short_lines as _restore_implausibly_short_lines_impl,
    _should_rollback_short_line_degradation as _should_rollback_short_line_degradation_impl,
)
from .whisper_integration_finalize import (
    _apply_low_quality_segment_postpasses as _apply_low_quality_segment_postpasses_impl,
    _finalize_whisper_line_set as _finalize_whisper_line_set_impl,
)
from .whisper_integration_hooks import (
    AlignmentPassHooks,
    CorrectTimingHooks,
    entrypoint_correct_timing_hook_kwargs,
)
from .whisper_runtime_config import WhisperRuntimeConfig

logger = get_logger(__name__)

_ALIASES = build_aliases()

__all__ = [
    "transcribe_vocals",
    "correct_timing_with_whisper",
    "align_lrc_text_to_whisper_timings",
]

_MISSING = object()


def _resolve_integration_attr(name: str):
    override = globals().get(name, _MISSING)
    if override is not _MISSING:
        return override
    return _ALIASES[name]


def _build_alignment_hooks() -> AlignmentPassHooks:
    return AlignmentPassHooks(
        transcribe_vocals_fn=transcribe_vocals,
        extract_audio_features_fn=extract_audio_features,
        dedupe_whisper_segments_fn=_resolve_integration_attr(
            "_dedupe_whisper_segments"
        ),
        trim_whisper_transcription_by_lyrics_fn=_resolve_integration_attr(
            "_trim_whisper_transcription_by_lyrics"
        ),
        fill_vocal_activity_gaps_fn=_resolve_integration_attr(
            "_fill_vocal_activity_gaps"
        ),
        dedupe_whisper_words_fn=_resolve_integration_attr("_dedupe_whisper_words"),
        extract_lrc_words_all_fn=_resolve_integration_attr("_extract_lrc_words_all"),
        build_phoneme_tokens_from_lrc_words_fn=_resolve_integration_attr(
            "_build_phoneme_tokens_from_lrc_words"
        ),
        build_phoneme_tokens_from_whisper_words_fn=_resolve_integration_attr(
            "_build_phoneme_tokens_from_whisper_words"
        ),
        build_syllable_tokens_from_phonemes_fn=_resolve_integration_attr(
            "_build_syllable_tokens_from_phonemes"
        ),
        build_segment_text_overlap_assignments_fn=_resolve_integration_attr(
            "_build_segment_text_overlap_assignments"
        ),
        build_phoneme_dtw_path_fn=_resolve_integration_attr("_build_phoneme_dtw_path"),
        build_word_assignments_from_phoneme_path_fn=_resolve_integration_attr(
            "_build_word_assignments_from_phoneme_path"
        ),
        build_block_segmented_syllable_assignments_fn=_resolve_integration_attr(
            "_build_block_segmented_syllable_assignments"
        ),
        map_lrc_words_to_whisper_fn=_resolve_integration_attr(
            "_map_lrc_words_to_whisper"
        ),
        shift_repeated_lines_to_next_whisper_fn=_resolve_integration_attr(
            "_shift_repeated_lines_to_next_whisper"
        ),
        enforce_monotonic_line_starts_whisper_fn=_resolve_integration_attr(
            "_enforce_monotonic_line_starts_whisper"
        ),
        resolve_line_overlaps_fn=_resolve_integration_attr("_resolve_line_overlaps"),
        extend_line_to_trailing_whisper_matches_fn=_resolve_integration_attr(
            "_extend_line_to_trailing_whisper_matches"
        ),
        pull_late_lines_to_matching_segments_fn=_resolve_integration_attr(
            "_pull_late_lines_to_matching_segments"
        ),
        retime_short_interjection_lines_fn=_resolve_integration_attr(
            "_retime_short_interjection_lines"
        ),
        snap_first_word_to_whisper_onset_fn=_resolve_integration_attr(
            "_snap_first_word_to_whisper_onset"
        ),
        interpolate_unmatched_lines_fn=_resolve_integration_attr(
            "_interpolate_unmatched_lines"
        ),
        refine_unmatched_lines_with_onsets_fn=_resolve_integration_attr(
            "_refine_unmatched_lines_with_onsets"
        ),
        pull_lines_forward_for_continuous_vocals_fn=_resolve_integration_attr(
            "_pull_lines_forward_for_continuous_vocals"
        ),
        normalize_line_word_timings_fn=_resolve_integration_attr(
            "_normalize_line_word_timings"
        ),
        enforce_monotonic_line_starts_fn=_resolve_integration_attr(
            "_enforce_monotonic_line_starts"
        ),
        enforce_non_overlapping_lines_fn=_resolve_integration_attr(
            "_enforce_non_overlapping_lines"
        ),
    )


def _build_correct_timing_hooks() -> CorrectTimingHooks:
    return CorrectTimingHooks(
        transcribe_vocals_fn=transcribe_vocals,
        extract_audio_features_fn=extract_audio_features,
        trim_whisper_transcription_by_lyrics_fn=_resolve_integration_attr(
            "_trim_whisper_transcription_by_lyrics"
        ),
        fill_vocal_activity_gaps_fn=_resolve_integration_attr(
            "_fill_vocal_activity_gaps"
        ),
        assess_lrc_quality_fn=_resolve_integration_attr("_assess_lrc_quality"),
        align_hybrid_lrc_whisper_fn=_resolve_integration_attr(
            "align_hybrid_lrc_whisper"
        ),
        align_dtw_whisper_with_data_fn=_resolve_integration_attr(
            "_align_dtw_whisper_with_data"
        ),
        retime_lines_from_dtw_alignments_fn=_resolve_integration_attr(
            "_retime_lines_from_dtw_alignments"
        ),
        apply_low_quality_segment_postpasses_fn=(
            _apply_low_quality_segment_postpasses_impl
        ),
        finalize_whisper_line_set_fn=_finalize_whisper_line_set_impl,
        constrain_line_starts_to_baseline_fn=(_constrain_line_starts_to_baseline_impl),
        should_rollback_short_line_degradation_fn=(
            _should_rollback_short_line_degradation_impl
        ),
        restore_implausibly_short_lines_fn=_restore_implausibly_short_lines_impl,
        clone_lines_for_fallback_fn=_clone_lines_for_fallback_impl,
        merge_first_two_lines_if_segment_matches_fn=_resolve_integration_attr(
            "_merge_first_two_lines_if_segment_matches"
        ),
        retime_adjacent_lines_to_whisper_window_fn=_resolve_integration_attr(
            "_retime_adjacent_lines_to_whisper_window"
        ),
        retime_adjacent_lines_to_segment_window_fn=_resolve_integration_attr(
            "_retime_adjacent_lines_to_segment_window"
        ),
        pull_next_line_into_segment_window_fn=_resolve_integration_attr(
            "_pull_next_line_into_segment_window"
        ),
        pull_lines_near_segment_end_fn=_resolve_integration_attr(
            "_pull_lines_near_segment_end"
        ),
        pull_next_line_into_same_segment_fn=_resolve_integration_attr(
            "_pull_next_line_into_same_segment"
        ),
        merge_lines_to_whisper_segments_fn=_resolve_integration_attr(
            "_merge_lines_to_whisper_segments"
        ),
        tighten_lines_to_whisper_segments_fn=_resolve_integration_attr(
            "_tighten_lines_to_whisper_segments"
        ),
        pull_lines_to_best_segments_fn=_resolve_integration_attr(
            "_pull_lines_to_best_segments"
        ),
        fix_ordering_violations_fn=_resolve_integration_attr(
            "_fix_ordering_violations"
        ),
        normalize_line_word_timings_fn=_resolve_integration_attr(
            "_normalize_line_word_timings"
        ),
        enforce_monotonic_line_starts_fn=_resolve_integration_attr(
            "_enforce_monotonic_line_starts"
        ),
        enforce_non_overlapping_lines_fn=_resolve_integration_attr(
            "_enforce_non_overlapping_lines"
        ),
        merge_short_following_line_into_segment_fn=_resolve_integration_attr(
            "_merge_short_following_line_into_segment"
        ),
        clamp_repeated_line_duration_fn=_resolve_integration_attr(
            "_clamp_repeated_line_duration"
        ),
        drop_duplicate_lines_fn=_resolve_integration_attr("_drop_duplicate_lines"),
        drop_duplicate_lines_by_timing_fn=_resolve_integration_attr(
            "_drop_duplicate_lines_by_timing"
        ),
        pull_lines_forward_for_continuous_vocals_fn=_resolve_integration_attr(
            "_pull_lines_forward_for_continuous_vocals"
        ),
    )


def _set_global_overrides(overrides: dict[str, object]) -> dict[str, object]:
    previous: dict[str, object] = {}
    module_globals = globals()
    for name, new_value in overrides.items():
        previous[name] = module_globals.get(name, _MISSING)
        module_globals[name] = new_value
    return previous


def _restore_global_overrides(previous: dict[str, object]) -> None:
    module_globals = globals()
    for name, value in previous.items():
        if value is _MISSING:
            module_globals.pop(name, None)
            continue
        module_globals[name] = value


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
    requested_overrides = {
        "transcribe_vocals": transcribe_vocals_fn,
        "extract_audio_features": extract_audio_features_fn,
        "_assess_lrc_quality": assess_lrc_quality_fn,
        "align_hybrid_lrc_whisper": align_hybrid_lrc_whisper_fn,
        "_align_dtw_whisper_with_data": align_dtw_whisper_with_data_fn,
        "_load_whisper_model_class": load_whisper_model_class_fn,
        "_get_whisper_cache_path": get_whisper_cache_path_fn,
        "_load_whisper_cache": load_whisper_cache_fn,
        "_save_whisper_cache": save_whisper_cache_fn,
        "_retime_lines_from_dtw_alignments": retime_lines_from_dtw_alignments_fn,
        "_get_ipa": get_ipa_fn,
    }
    active_overrides = {
        name: value for name, value in requested_overrides.items() if value is not None
    }
    previous = _set_global_overrides(active_overrides)

    try:
        yield
    finally:
        _restore_global_overrides(previous)


def _load_whisper_model_class():
    from faster_whisper import WhisperModel  # type: ignore

    return WhisperModel


def transcribe_vocals(
    vocals_path: str,
    language: Optional[str] = None,
    model_size: str = "base",
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
        get_whisper_cache_path_fn=_resolve_integration_attr("_get_whisper_cache_path"),
        find_best_cached_whisper_model_fn=_resolve_integration_attr(
            "_find_best_cached_whisper_model"
        ),
        load_whisper_cache_fn=_resolve_integration_attr("_load_whisper_cache"),
        save_whisper_cache_fn=_resolve_integration_attr("_save_whisper_cache"),
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
    runtime_config: Optional[WhisperRuntimeConfig] = None,
) -> Tuple[List[models.Line], List[str], Dict[str, float]]:
    """Align LRC text to Whisper timings using phonetic DTW."""
    hooks = _build_alignment_hooks()
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
        runtime_config=runtime_config,
        transcribe_vocals_fn=hooks.transcribe_vocals_fn,
        extract_audio_features_fn=hooks.extract_audio_features_fn,
        dedupe_whisper_segments_fn=hooks.dedupe_whisper_segments_fn,
        trim_whisper_transcription_by_lyrics_fn=(
            hooks.trim_whisper_transcription_by_lyrics_fn
        ),
        fill_vocal_activity_gaps_fn=hooks.fill_vocal_activity_gaps_fn,
        dedupe_whisper_words_fn=hooks.dedupe_whisper_words_fn,
        extract_lrc_words_all_fn=hooks.extract_lrc_words_all_fn,
        build_phoneme_tokens_from_lrc_words_fn=(
            hooks.build_phoneme_tokens_from_lrc_words_fn
        ),
        build_phoneme_tokens_from_whisper_words_fn=(
            hooks.build_phoneme_tokens_from_whisper_words_fn
        ),
        build_syllable_tokens_from_phonemes_fn=(
            hooks.build_syllable_tokens_from_phonemes_fn
        ),
        build_segment_text_overlap_assignments_fn=(
            hooks.build_segment_text_overlap_assignments_fn
        ),
        build_phoneme_dtw_path_fn=hooks.build_phoneme_dtw_path_fn,
        build_word_assignments_from_phoneme_path_fn=(
            hooks.build_word_assignments_from_phoneme_path_fn
        ),
        build_block_segmented_syllable_assignments_fn=(
            hooks.build_block_segmented_syllable_assignments_fn
        ),
        map_lrc_words_to_whisper_fn=hooks.map_lrc_words_to_whisper_fn,
        shift_repeated_lines_to_next_whisper_fn=(
            hooks.shift_repeated_lines_to_next_whisper_fn
        ),
        enforce_monotonic_line_starts_whisper_fn=(
            hooks.enforce_monotonic_line_starts_whisper_fn
        ),
        resolve_line_overlaps_fn=hooks.resolve_line_overlaps_fn,
        extend_line_to_trailing_whisper_matches_fn=(
            hooks.extend_line_to_trailing_whisper_matches_fn
        ),
        pull_late_lines_to_matching_segments_fn=(
            hooks.pull_late_lines_to_matching_segments_fn
        ),
        retime_short_interjection_lines_fn=hooks.retime_short_interjection_lines_fn,
        snap_first_word_to_whisper_onset_fn=hooks.snap_first_word_to_whisper_onset_fn,
        interpolate_unmatched_lines_fn=hooks.interpolate_unmatched_lines_fn,
        refine_unmatched_lines_with_onsets_fn=(
            hooks.refine_unmatched_lines_with_onsets_fn
        ),
        pull_lines_forward_for_continuous_vocals_fn=(
            hooks.pull_lines_forward_for_continuous_vocals_fn
        ),
        normalize_line_word_timings_fn=hooks.normalize_line_word_timings_fn,
        enforce_monotonic_line_starts_fn=hooks.enforce_monotonic_line_starts_fn,
        enforce_non_overlapping_lines_fn=hooks.enforce_non_overlapping_lines_fn,
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
    runtime_config: Optional[WhisperRuntimeConfig] = None,
) -> Tuple[List[models.Line], List[str], Dict[str, float]]:
    """Correct lyrics timing using Whisper transcription (adaptive approach)."""
    hooks = _build_correct_timing_hooks()
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
        runtime_config=runtime_config,
        **entrypoint_correct_timing_hook_kwargs(hooks),
        logger=logger,
    )


def __getattr__(name: str):
    if name in _ALIASES:
        return _resolve_integration_attr(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_ALIASES.keys()))
