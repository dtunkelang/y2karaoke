import pytest

from y2karaoke.core.components.alignment.timing_models import (
    TranscriptionSegment,
    TranscriptionWord,
)
from y2karaoke.core.components.whisper import whisper_blocks
from y2karaoke.core.components.whisper import (
    whisper_segment_assignment_experimental as wsae,
)


def _sample_segments() -> list[TranscriptionSegment]:
    return [
        TranscriptionSegment(start=10.0, end=12.0, text="hello world", words=[]),
        TranscriptionSegment(start=13.0, end=15.0, text="good night", words=[]),
    ]


def _sample_words() -> list[TranscriptionWord]:
    return [
        TranscriptionWord(text="hello", start=10.1, end=10.5, probability=1.0),
        TranscriptionWord(text="world", start=10.5, end=11.1, probability=1.0),
        TranscriptionWord(text="good", start=13.1, end=13.6, probability=1.0),
        TranscriptionWord(text="night", start=13.6, end=14.1, probability=1.0),
    ]


def _sample_lrc_words() -> list[dict]:
    return [
        {"line_idx": 0, "text": "hello"},
        {"line_idx": 0, "text": "world"},
        {"line_idx": 1, "text": "good"},
        {"line_idx": 1, "text": "night"},
    ]


def test_parallel_assigner_defaults_to_current_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("Y2K_WHISPER_SEGMENT_ASSIGN_PIPELINE", raising=False)

    experimental = wsae.build_segment_text_overlap_assignments(
        _sample_lrc_words(), _sample_words(), _sample_segments()
    )
    baseline = whisper_blocks._build_segment_text_overlap_assignments(
        _sample_lrc_words(), _sample_words(), _sample_segments()
    )

    assert dict(experimental) == dict(baseline)


def test_parallel_assigner_experimental_mode_preserves_current_behavior_initially(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("Y2K_WHISPER_SEGMENT_ASSIGN_PIPELINE", "parallel_experimental")

    experimental = wsae.build_segment_text_overlap_assignments(
        _sample_lrc_words(), _sample_words(), _sample_segments()
    )
    baseline = whisper_blocks._build_segment_text_overlap_assignments(
        _sample_lrc_words(), _sample_words(), _sample_segments()
    )

    assert dict(experimental) == dict(baseline)
