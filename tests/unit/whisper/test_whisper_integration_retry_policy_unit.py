from y2karaoke.core.components.whisper.whisper_integration_retry import (
    retry_improves_alignment,
    should_retry_with_aggressive_whisper,
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
