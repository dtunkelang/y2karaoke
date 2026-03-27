from tools import analyze_advisory_support_candidates as tool
from y2karaoke.core.components.whisper.whisper_advisory_support import (
    AdvisoryLineSupport,
)


def test_candidate_bucket_marks_high_confidence_exact_gain() -> None:
    bucket = tool.advisory_candidate_bucket(
        AdvisoryLineSupport(
            index=3,
            text="I've been inclined",
            current_window_word_count=2,
            default_window_word_count=0,
            aggressive_window_word_count=4,
            default_best_segment_text="",
            aggressive_best_segment_text="I've been inclined",
            default_best_overlap=0.0,
            aggressive_best_overlap=1.0,
            aggressive_gain=True,
        )
    )

    assert bucket == "high_confidence"


def test_candidate_bucket_rejects_weak_merged_overlap() -> None:
    bucket = tool.advisory_candidate_bucket(
        AdvisoryLineSupport(
            index=2,
            text="Take me on",
            current_window_word_count=0,
            default_window_word_count=0,
            aggressive_window_word_count=4,
            default_best_segment_text="",
            aggressive_best_segment_text="noise words",
            default_best_overlap=0.0,
            aggressive_best_overlap=0.273,
            aggressive_gain=False,
        )
    )

    assert bucket is None


def test_candidate_score_prefers_strong_aggressive_overlap() -> None:
    strong = tool.advisory_candidate_score(
        AdvisoryLineSupport(
            index=3,
            text="I've been inclined",
            current_window_word_count=0,
            default_window_word_count=0,
            aggressive_window_word_count=4,
            default_best_segment_text="",
            aggressive_best_segment_text="I've been inclined",
            default_best_overlap=0.0,
            aggressive_best_overlap=1.0,
            aggressive_gain=True,
        )
    )
    weak = tool.advisory_candidate_score(
        AdvisoryLineSupport(
            index=3,
            text="I've been inclined",
            current_window_word_count=0,
            default_window_word_count=0,
            aggressive_window_word_count=4,
            default_best_segment_text="",
            aggressive_best_segment_text="I've been inclined",
            default_best_overlap=0.0,
            aggressive_best_overlap=0.45,
            aggressive_gain=False,
        )
    )

    assert strong > weak
