import pytest

from y2karaoke.core.models import Line, Word
import y2karaoke.core.components.whisper.whisper_integration as wi
from y2karaoke.core.components.whisper import whisper_integration_transcribe as witx
from y2karaoke.core.components.whisper.whisper_integration_baseline import (
    _should_rollback_short_line_degradation,
)
from y2karaoke.core.components.alignment.timing_models import (
    TranscriptionSegment,
    TranscriptionWord,
)


def test_should_rollback_short_line_degradation_triggers():
    original = [
        Line(
            words=[
                Word(text="a", start_time=0.0, end_time=0.4),
                Word(text="b", start_time=0.4, end_time=0.8),
                Word(text="c", start_time=0.8, end_time=1.2),
            ]
        )
        for _ in range(12)
    ]
    degraded = [
        Line(
            words=[
                Word(text="a", start_time=0.0, end_time=0.05),
                Word(text="b", start_time=0.05, end_time=0.1),
                Word(text="c", start_time=0.1, end_time=0.15),
            ]
        )
        for _ in range(12)
    ]

    rollback, before, after = _should_rollback_short_line_degradation(
        original, degraded
    )

    assert rollback
    assert before == 0
    assert after == 12


def test_should_rollback_short_line_degradation_ignores_small_change():
    original = [
        Line(
            words=[
                Word(text="a", start_time=0.0, end_time=0.2),
                Word(text="b", start_time=0.2, end_time=0.4),
                Word(text="c", start_time=0.4, end_time=0.6),
            ]
        )
        for _ in range(20)
    ]
    slightly_worse = list(original)
    slightly_worse[0] = Line(
        words=[
            Word(text="a", start_time=0.0, end_time=0.05),
            Word(text="b", start_time=0.05, end_time=0.1),
            Word(text="c", start_time=0.1, end_time=0.15),
        ]
    )
    slightly_worse[1] = Line(
        words=[
            Word(text="a", start_time=0.7, end_time=0.75),
            Word(text="b", start_time=0.75, end_time=0.8),
            Word(text="c", start_time=0.8, end_time=0.85),
        ]
    )

    rollback, before, after = _should_rollback_short_line_degradation(
        original, slightly_worse
    )

    assert not rollback
    assert before == 0
    assert after == 2


def test_should_accept_whisperx_upgrade_accepts_sane_shape():
    base_words = [
        TranscriptionWord(text="a", start=0.2, end=0.4, probability=0.9),
        TranscriptionWord(text="b", start=8.0, end=8.2, probability=0.9),
    ]
    base_segments = [TranscriptionSegment(start=0.2, end=8.2, text="ab", words=[])]
    upgraded_words = [
        witx._WhisperxWord(
            start=float(i) * 0.12,
            end=float(i) * 0.12 + 0.08,
            text=f"w{i}",
            probability=0.9,
        )
        for i in range(120)
    ]
    upgraded_segments = [
        witx._WhisperxSegment(
            start=float(i),
            end=float(i) + 0.9,
            text=f"s{i}",
            words=upgraded_words[i * 20 : (i + 1) * 20],
        )
        for i in range(6)
    ]

    accepted = witx._should_accept_whisperx_upgrade(
        base_segments=base_segments,
        base_words=base_words,
        upgraded_segments=upgraded_segments,
        upgraded_words=upgraded_words,
        logger=wi.logger,
    )

    assert accepted is True


def test_should_accept_whisperx_upgrade_rejects_excessive_overlap():
    base_words = [
        TranscriptionWord(text="a", start=0.0, end=0.2, probability=0.9),
        TranscriptionWord(text="b", start=5.0, end=5.2, probability=0.9),
    ]
    base_segments = [TranscriptionSegment(start=0.0, end=5.2, text="ab", words=[])]
    upgraded_words = []
    for i in range(120):
        if i == 0:
            start = 0.0
        else:
            start = upgraded_words[-1].start + 0.03
        end = start + 0.08
        if i % 10 == 0 and i > 0:
            start -= 0.08
            end -= 0.08
        upgraded_words.append(
            witx._WhisperxWord(start=start, end=end, text=f"w{i}", probability=0.9)
        )
    upgraded_segments = [
        witx._WhisperxSegment(
            start=float(i),
            end=float(i) + 1.0,
            text=f"s{i}",
            words=upgraded_words[i * 20 : (i + 1) * 20],
        )
        for i in range(6)
    ]

    accepted = witx._should_accept_whisperx_upgrade(
        base_segments=base_segments,
        base_words=base_words,
        upgraded_segments=upgraded_segments,
        upgraded_words=upgraded_words,
        logger=wi.logger,
    )

    assert accepted is False


def test_should_accept_whisperx_upgrade_rejects_shorter_span():
    base_words = [
        TranscriptionWord(text="a", start=2.0, end=2.2, probability=0.9),
        TranscriptionWord(text="b", start=20.0, end=20.4, probability=0.9),
    ]
    base_segments = [TranscriptionSegment(start=2.0, end=20.4, text="ab", words=[])]
    upgraded_words = [
        witx._WhisperxWord(
            start=float(i) * 0.1,
            end=float(i) * 0.1 + 0.07,
            text=f"w{i}",
            probability=0.9,
        )
        for i in range(120)
    ]
    upgraded_segments = [
        witx._WhisperxSegment(
            start=float(i) * 2.0,
            end=float(i) * 2.0 + 1.0,
            text=f"s{i}",
            words=upgraded_words[i * 20 : (i + 1) * 20],
        )
        for i in range(6)
    ]

    accepted = witx._should_accept_whisperx_upgrade(
        base_segments=base_segments,
        base_words=base_words,
        upgraded_segments=upgraded_segments,
        upgraded_words=upgraded_words,
        logger=wi.logger,
    )

    assert accepted is False


def test_normalize_whisperx_segments_enforces_monotonic_words():
    segments = [
        witx._WhisperxSegment(
            start=0.0,
            end=1.0,
            text="a",
            words=[
                witx._WhisperxWord(start=0.10, end=0.25, text="w1", probability=0.9),
                witx._WhisperxWord(start=0.21, end=0.30, text="w2", probability=0.9),
            ],
        ),
        witx._WhisperxSegment(
            start=0.9,
            end=1.3,
            text="b",
            words=[
                witx._WhisperxWord(start=0.28, end=0.40, text="w3", probability=0.9),
            ],
        ),
    ]

    normalized_segments, normalized_words = witx._normalize_whisperx_segments(segments)

    assert len(normalized_segments) == 2
    assert len(normalized_words) == 3
    starts = [w.start for w in normalized_words]
    ends = [w.end for w in normalized_words]
    assert starts == sorted(starts)
    assert all(ends[i] <= starts[i + 1] for i in range(len(starts) - 1))
    assert normalized_segments[0].start == pytest.approx(normalized_words[0].start)
    assert normalized_segments[-1].end == pytest.approx(normalized_words[-1].end)


def test_default_transcription_config_variants():
    base = witx._default_transcription_config(aggressive=False)
    aggr = witx._default_transcription_config(aggressive=True)

    assert base.use_vad_filter is True
    assert base.no_speech_threshold is None
    assert base.log_prob_threshold is None
    assert aggr.use_vad_filter is False
    assert aggr.no_speech_threshold == 1.0
    assert aggr.log_prob_threshold == -2.0


def test_run_whisper_transcription_applies_aggressive_kwargs():
    class DummyModel:
        def __init__(self):
            self.calls = []

        def transcribe(self, vocals_path, **kwargs):
            self.calls.append((vocals_path, kwargs))
            return "segments", "info"

    model = DummyModel()

    segments, info = witx._run_whisper_transcription(
        model=model,
        vocals_path="vocals.wav",
        language="en",
        aggressive=True,
        temperature=0.3,
    )

    assert segments == "segments"
    assert info == "info"
    assert len(model.calls) == 1
    call_path, call_kwargs = model.calls[0]
    assert call_path == "vocals.wav"
    assert call_kwargs["language"] == "en"
    assert call_kwargs["word_timestamps"] is True
    assert call_kwargs["vad_filter"] is False
    assert call_kwargs["temperature"] == 0.3
    assert call_kwargs["no_speech_threshold"] == 1.0
    assert call_kwargs["log_prob_threshold"] == -2.0
