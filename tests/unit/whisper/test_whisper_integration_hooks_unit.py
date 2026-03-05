from y2karaoke.core.components.whisper.whisper_integration_hooks import (
    CorrectTimingHooks,
    correct_timing_hook_kwargs,
)


def test_correct_timing_hook_kwargs_maps_selected_hooks():
    def _stub(*_args, **_kwargs):
        return None

    hooks = CorrectTimingHooks(
        transcribe_vocals_fn=_stub,
        extract_audio_features_fn=_stub,
        trim_whisper_transcription_by_lyrics_fn=_stub,
        fill_vocal_activity_gaps_fn=_stub,
        assess_lrc_quality_fn=_stub,
        align_hybrid_lrc_whisper_fn=_stub,
        align_dtw_whisper_with_data_fn=_stub,
        retime_lines_from_dtw_alignments_fn=_stub,
        apply_low_quality_segment_postpasses_fn=_stub,
        finalize_whisper_line_set_fn=_stub,
        constrain_line_starts_to_baseline_fn=_stub,
        should_rollback_short_line_degradation_fn=_stub,
        restore_implausibly_short_lines_fn=_stub,
        clone_lines_for_fallback_fn=_stub,
        merge_first_two_lines_if_segment_matches_fn=_stub,
        retime_adjacent_lines_to_whisper_window_fn=_stub,
        retime_adjacent_lines_to_segment_window_fn=_stub,
        pull_next_line_into_segment_window_fn=_stub,
        pull_lines_near_segment_end_fn=_stub,
        pull_next_line_into_same_segment_fn=_stub,
        merge_lines_to_whisper_segments_fn=_stub,
        tighten_lines_to_whisper_segments_fn=_stub,
        pull_lines_to_best_segments_fn=_stub,
        fix_ordering_violations_fn=_stub,
        normalize_line_word_timings_fn=_stub,
        enforce_monotonic_line_starts_fn=_stub,
        enforce_non_overlapping_lines_fn=_stub,
        merge_short_following_line_into_segment_fn=_stub,
        clamp_repeated_line_duration_fn=_stub,
        drop_duplicate_lines_fn=_stub,
        drop_duplicate_lines_by_timing_fn=_stub,
        pull_lines_forward_for_continuous_vocals_fn=_stub,
    )

    kwargs = correct_timing_hook_kwargs(hooks)

    assert len(kwargs) == 32
    assert kwargs["transcribe_vocals_fn"] is _stub
    assert kwargs["finalize_whisper_line_set_fn"] is _stub
    assert kwargs["drop_duplicate_lines_by_timing_fn"] is _stub
