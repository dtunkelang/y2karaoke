from y2karaoke.core.components.alignment.timing_models import (
    TranscriptionSegment,
    TranscriptionWord,
)
from tools.analyze_line_support_variants import _summarize_lines


def test_summarize_lines_delegates_to_shared_support_logic() -> None:
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
        default_segments=[],
        default_words=[],
        aggressive_segments=aggressive_segments,
        aggressive_words=aggressive_words,
    )

    assert summaries[0].aggressive_gain is True
