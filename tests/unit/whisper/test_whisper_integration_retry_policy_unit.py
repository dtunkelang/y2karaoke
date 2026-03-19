from y2karaoke.core.components.whisper.whisper_integration_retry import (
    retry_improves_alignment,
    should_retry_with_aggressive_whisper,
)
from y2karaoke.core.components.whisper.whisper_runtime_config import (
    WhisperRuntimeConfig,
)


def test_should_retry_with_aggressive_whisper_borderline_true():
    assert (
        should_retry_with_aggressive_whisper(
            line_count=40,
            aggressive=False,
            metrics={
                "matched_ratio": 0.82,
                "line_coverage": 0.86,
                "phonetic_similarity_coverage": 0.4,
            },
        )
        is True
    )


def test_should_retry_with_aggressive_whisper_non_borderline_false():
    assert (
        should_retry_with_aggressive_whisper(
            line_count=40,
            aggressive=False,
            metrics={
                "matched_ratio": 0.65,
                "line_coverage": 0.9,
                "phonetic_similarity_coverage": 0.5,
            },
        )
        is False
    )


def test_should_retry_with_aggressive_whisper_on_empty_transcription_short_clip():
    assert (
        should_retry_with_aggressive_whisper(
            line_count=4,
            aggressive=False,
            metrics={"transcription_empty": 1.0},
        )
        is True
    )


def test_retry_improves_alignment_true_on_coverage_gain():
    assert (
        retry_improves_alignment(
            {
                "matched_ratio": 0.8,
                "line_coverage": 0.84,
                "phonetic_similarity_coverage": 0.4,
            },
            {
                "matched_ratio": 0.8,
                "line_coverage": 0.9,
                "phonetic_similarity_coverage": 0.41,
            },
        )
        is True
    )


def test_retry_improves_alignment_false_on_tiny_gain():
    assert (
        retry_improves_alignment(
            {
                "matched_ratio": 0.8,
                "line_coverage": 0.84,
                "phonetic_similarity_coverage": 0.4,
            },
            {
                "matched_ratio": 0.801,
                "line_coverage": 0.841,
                "phonetic_similarity_coverage": 0.401,
            },
        )
        is False
    )


def test_retry_improves_alignment_profile_safe_is_stricter(monkeypatch):
    baseline = {
        "matched_ratio": 0.8,
        "line_coverage": 0.84,
        "phonetic_similarity_coverage": 0.4,
    }
    retry = {
        "matched_ratio": 0.833,
        "line_coverage": 0.83,
        "phonetic_similarity_coverage": 0.44,
    }

    monkeypatch.setenv("Y2K_WHISPER_PROFILE", "default")
    assert retry_improves_alignment(baseline, retry) is True

    monkeypatch.setenv("Y2K_WHISPER_PROFILE", "safe")
    assert retry_improves_alignment(baseline, retry) is False


def test_retry_improves_alignment_profile_can_be_passed_explicitly():
    baseline = {
        "matched_ratio": 0.8,
        "line_coverage": 0.84,
        "phonetic_similarity_coverage": 0.4,
    }
    retry = {
        "matched_ratio": 0.833,
        "line_coverage": 0.83,
        "phonetic_similarity_coverage": 0.44,
    }

    assert retry_improves_alignment(
        baseline,
        retry,
        runtime_config=WhisperRuntimeConfig(profile="default"),
    )
    assert not retry_improves_alignment(
        baseline,
        retry,
        runtime_config=WhisperRuntimeConfig(profile="safe"),
    )
