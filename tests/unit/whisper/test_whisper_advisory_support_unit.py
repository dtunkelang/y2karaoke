from y2karaoke.core.components.alignment.timing_models import (
    TranscriptionSegment,
    TranscriptionWord,
)
from y2karaoke.core.components.whisper.whisper_advisory_support import (
    AdvisoryLineSupport,
    advisory_candidate_bucket,
    advisory_candidate_score,
    summarize_line_support,
)


def test_summarize_line_support_flags_aggressive_gain() -> None:
    report_lines = [
        {
            "index": 1,
            "text": "I've been inclined",
            "start": 12.74,
            "end": 14.12,
            "whisper_window_word_count": 0,
        }
    ]
    default_segments = [
        TranscriptionSegment(start=8.72, end=25.18, text="The night", words=[])
    ]
    default_words: list[TranscriptionWord] = []
    aggressive_words = [
        TranscriptionWord(text="I've", start=12.0, end=12.4, probability=0.9),
        TranscriptionWord(text="been", start=12.4, end=12.8, probability=0.9),
        TranscriptionWord(text="inclined", start=12.8, end=13.8, probability=0.9),
    ]
    aggressive_segments = [
        TranscriptionSegment(
            start=12.0,
            end=13.8,
            text="I've been inclined",
            words=aggressive_words,
        )
    ]

    summaries = summarize_line_support(
        report_lines=report_lines,
        default_segments=default_segments,
        default_words=default_words,
        aggressive_segments=aggressive_segments,
        aggressive_words=aggressive_words,
    )

    assert len(summaries) == 1
    assert summaries[0].aggressive_gain is True
    assert summaries[0].aggressive_window_word_count == 3
    assert summaries[0].aggressive_best_overlap == 1.0


def test_summarize_line_support_ignores_weak_overlap() -> None:
    report_lines = [
        {
            "index": 1,
            "text": "Take me on",
            "start": 6.84,
            "end": 10.42,
            "whisper_window_word_count": 0,
        }
    ]
    aggressive_words = [
        TranscriptionWord(text="noise", start=7.0, end=7.4, probability=0.9),
        TranscriptionWord(text="words", start=7.4, end=7.8, probability=0.9),
    ]
    aggressive_segments = [
        TranscriptionSegment(
            start=7.0,
            end=7.8,
            text="noise words",
            words=aggressive_words,
        )
    ]

    summaries = summarize_line_support(
        report_lines=report_lines,
        default_segments=[],
        default_words=[],
        aggressive_segments=aggressive_segments,
        aggressive_words=aggressive_words,
    )

    assert summaries[0].aggressive_gain is False


def test_advisory_candidate_bucket_marks_high_confidence_exact_gain() -> None:
    summary = AdvisoryLineSupport(
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

    assert advisory_candidate_bucket(summary) == "high_confidence"


def test_advisory_candidate_bucket_allows_exact_gain_with_extra_window_word() -> None:
    summary = AdvisoryLineSupport(
        index=3,
        text="I've been inclined",
        current_window_word_count=4,
        default_window_word_count=4,
        aggressive_window_word_count=4,
        default_best_segment_text="merged default segment",
        aggressive_best_segment_text="I've been inclined",
        default_best_overlap=0.13,
        aggressive_best_overlap=1.0,
        aggressive_gain=False,
    )

    assert advisory_candidate_bucket(summary) == "medium_confidence"


def test_advisory_candidate_bucket_rejects_weak_merged_overlap() -> None:
    summary = AdvisoryLineSupport(
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

    assert advisory_candidate_bucket(summary) is None


def test_advisory_candidate_score_prefers_strong_aggressive_overlap() -> None:
    strong = AdvisoryLineSupport(
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
    weak = AdvisoryLineSupport(
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

    assert advisory_candidate_score(strong) > advisory_candidate_score(weak)
