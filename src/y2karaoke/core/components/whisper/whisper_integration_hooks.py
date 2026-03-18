"""Hook bundles for Whisper integration pipeline dependency injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from ..alignment import timing_models


@dataclass(frozen=True)
class AlignmentPassHooks:
    transcribe_vocals_fn: Callable[..., Tuple[Any, Any, str, str]]
    extract_audio_features_fn: Callable[..., timing_models.AudioFeatures | None]
    dedupe_whisper_segments_fn: Callable[..., Any]
    trim_whisper_transcription_by_lyrics_fn: Callable[..., Any]
    fill_vocal_activity_gaps_fn: Callable[..., Any]
    dedupe_whisper_words_fn: Callable[..., Any]
    extract_lrc_words_all_fn: Callable[..., Any]
    build_phoneme_tokens_from_lrc_words_fn: Callable[..., Any]
    build_phoneme_tokens_from_whisper_words_fn: Callable[..., Any]
    build_syllable_tokens_from_phonemes_fn: Callable[..., Any]
    build_segment_text_overlap_assignments_fn: Callable[..., Any]
    build_phoneme_dtw_path_fn: Callable[..., Any]
    build_word_assignments_from_phoneme_path_fn: Callable[..., Any]
    build_block_segmented_syllable_assignments_fn: Callable[..., Any]
    map_lrc_words_to_whisper_fn: Callable[..., Any]
    shift_repeated_lines_to_next_whisper_fn: Callable[..., Any]
    enforce_monotonic_line_starts_whisper_fn: Callable[..., Any]
    resolve_line_overlaps_fn: Callable[..., Any]
    extend_line_to_trailing_whisper_matches_fn: Callable[..., Any]
    pull_late_lines_to_matching_segments_fn: Callable[..., Any]
    retime_short_interjection_lines_fn: Callable[..., Any]
    snap_first_word_to_whisper_onset_fn: Callable[..., Any]
    interpolate_unmatched_lines_fn: Callable[..., Any]
    refine_unmatched_lines_with_onsets_fn: Callable[..., Any]
    pull_lines_forward_for_continuous_vocals_fn: Callable[..., Any]


@dataclass(frozen=True)
class CorrectTimingHooks:
    transcribe_vocals_fn: Callable[..., Tuple[Any, Any, str, str]]
    extract_audio_features_fn: Callable[..., timing_models.AudioFeatures | None]
    trim_whisper_transcription_by_lyrics_fn: Callable[..., Any]
    fill_vocal_activity_gaps_fn: Callable[..., Any]
    assess_lrc_quality_fn: Callable[..., Any]
    align_hybrid_lrc_whisper_fn: Callable[..., Any]
    align_dtw_whisper_with_data_fn: Callable[..., Any]
    retime_lines_from_dtw_alignments_fn: Callable[..., Any]
    apply_low_quality_segment_postpasses_fn: Callable[..., Any]
    finalize_whisper_line_set_fn: Callable[..., Any]
    constrain_line_starts_to_baseline_fn: Callable[..., Any]
    should_rollback_short_line_degradation_fn: Callable[..., Any]
    restore_implausibly_short_lines_fn: Callable[..., Any]
    clone_lines_for_fallback_fn: Callable[..., Any]
    merge_first_two_lines_if_segment_matches_fn: Callable[..., Any]
    retime_adjacent_lines_to_whisper_window_fn: Callable[..., Any]
    retime_adjacent_lines_to_segment_window_fn: Callable[..., Any]
    pull_next_line_into_segment_window_fn: Callable[..., Any]
    pull_lines_near_segment_end_fn: Callable[..., Any]
    pull_next_line_into_same_segment_fn: Callable[..., Any]
    merge_lines_to_whisper_segments_fn: Callable[..., Any]
    tighten_lines_to_whisper_segments_fn: Callable[..., Any]
    pull_lines_to_best_segments_fn: Callable[..., Any]
    fix_ordering_violations_fn: Callable[..., Any]
    normalize_line_word_timings_fn: Callable[..., Any]
    enforce_monotonic_line_starts_fn: Callable[..., Any]
    enforce_non_overlapping_lines_fn: Callable[..., Any]
    merge_short_following_line_into_segment_fn: Callable[..., Any]
    clamp_repeated_line_duration_fn: Callable[..., Any]
    drop_duplicate_lines_fn: Callable[..., Any]
    drop_duplicate_lines_by_timing_fn: Callable[..., Any]
    pull_lines_forward_for_continuous_vocals_fn: Callable[..., Any]


def correct_timing_hook_kwargs(hooks: CorrectTimingHooks) -> Dict[str, Any]:
    return {
        "transcribe_vocals_fn": hooks.transcribe_vocals_fn,
        "extract_audio_features_fn": hooks.extract_audio_features_fn,
        "trim_whisper_transcription_by_lyrics_fn": (
            hooks.trim_whisper_transcription_by_lyrics_fn
        ),
        "fill_vocal_activity_gaps_fn": hooks.fill_vocal_activity_gaps_fn,
        "assess_lrc_quality_fn": hooks.assess_lrc_quality_fn,
        "align_hybrid_lrc_whisper_fn": hooks.align_hybrid_lrc_whisper_fn,
        "align_dtw_whisper_with_data_fn": hooks.align_dtw_whisper_with_data_fn,
        "retime_lines_from_dtw_alignments_fn": hooks.retime_lines_from_dtw_alignments_fn,
        "apply_low_quality_segment_postpasses_fn": (
            hooks.apply_low_quality_segment_postpasses_fn
        ),
        "finalize_whisper_line_set_fn": hooks.finalize_whisper_line_set_fn,
        "constrain_line_starts_to_baseline_fn": hooks.constrain_line_starts_to_baseline_fn,
        "should_rollback_short_line_degradation_fn": (
            hooks.should_rollback_short_line_degradation_fn
        ),
        "restore_implausibly_short_lines_fn": hooks.restore_implausibly_short_lines_fn,
        "clone_lines_for_fallback_fn": hooks.clone_lines_for_fallback_fn,
        "merge_first_two_lines_if_segment_matches_fn": (
            hooks.merge_first_two_lines_if_segment_matches_fn
        ),
        "retime_adjacent_lines_to_whisper_window_fn": (
            hooks.retime_adjacent_lines_to_whisper_window_fn
        ),
        "retime_adjacent_lines_to_segment_window_fn": (
            hooks.retime_adjacent_lines_to_segment_window_fn
        ),
        "pull_next_line_into_segment_window_fn": hooks.pull_next_line_into_segment_window_fn,
        "pull_lines_near_segment_end_fn": hooks.pull_lines_near_segment_end_fn,
        "pull_next_line_into_same_segment_fn": hooks.pull_next_line_into_same_segment_fn,
        "merge_lines_to_whisper_segments_fn": hooks.merge_lines_to_whisper_segments_fn,
        "tighten_lines_to_whisper_segments_fn": hooks.tighten_lines_to_whisper_segments_fn,
        "pull_lines_to_best_segments_fn": hooks.pull_lines_to_best_segments_fn,
        "fix_ordering_violations_fn": hooks.fix_ordering_violations_fn,
        "normalize_line_word_timings_fn": hooks.normalize_line_word_timings_fn,
        "enforce_monotonic_line_starts_fn": hooks.enforce_monotonic_line_starts_fn,
        "enforce_non_overlapping_lines_fn": hooks.enforce_non_overlapping_lines_fn,
        "merge_short_following_line_into_segment_fn": (
            hooks.merge_short_following_line_into_segment_fn
        ),
        "clamp_repeated_line_duration_fn": hooks.clamp_repeated_line_duration_fn,
        "drop_duplicate_lines_fn": hooks.drop_duplicate_lines_fn,
        "drop_duplicate_lines_by_timing_fn": hooks.drop_duplicate_lines_by_timing_fn,
        "pull_lines_forward_for_continuous_vocals_fn": (
            hooks.pull_lines_forward_for_continuous_vocals_fn
        ),
    }


def entrypoint_correct_timing_hook_kwargs(hooks: CorrectTimingHooks) -> Dict[str, Any]:
    kwargs = correct_timing_hook_kwargs(hooks)
    for key in (
        "apply_low_quality_segment_postpasses_fn",
        "finalize_whisper_line_set_fn",
        "constrain_line_starts_to_baseline_fn",
        "should_rollback_short_line_degradation_fn",
        "restore_implausibly_short_lines_fn",
        "clone_lines_for_fallback_fn",
    ):
        kwargs.pop(key)
    return kwargs
