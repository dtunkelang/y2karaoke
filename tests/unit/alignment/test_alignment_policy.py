from y2karaoke.core.components.alignment.alignment_policy import (
    decide_lrc_timing_trust,
    should_retry_aggressive_whisper_dtw_map,
)


def test_decide_lrc_timing_trust_keeps_moderate_mismatch():
    decision = decide_lrc_timing_trust(
        lrc_duration_mismatch_sec=10.0,
        can_recover_with_audio_alignment=True,
        likely_outro_padding=False,
    )
    assert decision.keep_lrc_timings is True
    assert decision.mode == "degraded_duration_mismatch"


def test_decide_lrc_timing_trust_keeps_outro_padding():
    decision = decide_lrc_timing_trust(
        lrc_duration_mismatch_sec=15.0,
        can_recover_with_audio_alignment=True,
        likely_outro_padding=True,
    )
    assert decision.keep_lrc_timings is True
    assert decision.mode == "degraded_outro_padding"


def test_decide_lrc_timing_trust_drops_only_when_recoverable_and_severe():
    decision = decide_lrc_timing_trust(
        lrc_duration_mismatch_sec=15.0,
        can_recover_with_audio_alignment=True,
        likely_outro_padding=False,
    )
    assert decision.keep_lrc_timings is False
    assert decision.mode == "dropped_duration_mismatch"


def test_should_retry_aggressive_whisper_dtw_map_true_for_borderline_case():
    assert (
        should_retry_aggressive_whisper_dtw_map(
            line_count=40,
            aggressive_already_enabled=False,
            metrics={
                "matched_ratio": 0.82,
                "line_coverage": 0.86,
                "phonetic_similarity_coverage": 0.4,
            },
        )
        is True
    )


def test_should_retry_aggressive_whisper_dtw_map_false_for_non_borderline_case():
    assert (
        should_retry_aggressive_whisper_dtw_map(
            line_count=40,
            aggressive_already_enabled=False,
            metrics={
                "matched_ratio": 0.65,
                "line_coverage": 0.9,
                "phonetic_similarity_coverage": 0.5,
            },
        )
        is False
    )
