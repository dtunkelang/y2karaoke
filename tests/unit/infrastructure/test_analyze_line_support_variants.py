from y2karaoke.core.components.alignment.timing_models import (
    TranscriptionSegment,
    TranscriptionWord,
)
from tools.analyze_line_support_variants import _summarize_lines


def test_summarize_lines_flags_aggressive_gain() -> None:
    report = {
        "lines": [
            {
                "index": 1,
                "text": "I've been inclined",
                "start": 12.74,
                "end": 14.12,
                "whisper_window_word_count": 0,
            }
        ]
    }
    default_segments = [
        TranscriptionSegment(start=8.72, end=25.18, text="The night", words=[])
    ]
    default_words = []
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

    summaries = _summarize_lines(
        report=report,
        default_segments=default_segments,
        default_words=default_words,
        aggressive_segments=aggressive_segments,
        aggressive_words=aggressive_words,
    )

    assert len(summaries) == 1
    assert summaries[0].aggressive_gain is True
    assert summaries[0].aggressive_window_word_count == 3
    assert summaries[0].aggressive_best_overlap == 1.0


def test_summarize_lines_ignores_aggressive_gain_without_overlap() -> None:
    report = {
        "lines": [
            {
                "index": 1,
                "text": "Take me on",
                "start": 6.84,
                "end": 10.42,
                "whisper_window_word_count": 0,
            }
        ]
    }
    default_segments = []
    default_words = []
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

    summaries = _summarize_lines(
        report=report,
        default_segments=default_segments,
        default_words=default_words,
        aggressive_segments=aggressive_segments,
        aggressive_words=aggressive_words,
    )

    assert summaries[0].aggressive_gain is False
