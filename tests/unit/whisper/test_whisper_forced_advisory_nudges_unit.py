from y2karaoke.core.components.alignment.timing_models import (
    TranscriptionSegment,
    TranscriptionWord,
)
from y2karaoke.core.components.whisper.whisper_forced_advisory_nudges import (
    apply_forced_advisory_start_nudges,
)
from y2karaoke.core.models import Line, Word


class _Logger:
    def info(self, *_args, **_kwargs):
        return None


def _line(text: str, start: float, end: float) -> Line:
    tokens = text.split()
    step = (end - start) / max(len(tokens), 1)
    return Line(
        words=[
            Word(
                text=token,
                start_time=start + idx * step,
                end_time=start + (idx + 1) * step,
            )
            for idx, token in enumerate(tokens)
        ]
    )


def test_apply_forced_advisory_start_nudges_shifts_only_exact_three_word_candidate(
    tmp_path,
) -> None:
    lines = [
        _line("Good times never seemed so good", 4.97, 8.60),
        _line("I've been inclined", 12.74, 14.12),
    ]
    current_segments = [
        TranscriptionSegment(
            start=4.97,
            end=8.60,
            text="Good times never seemed so good",
            words=[],
        )
    ]
    current_words = [
        TranscriptionWord(text="They", start=9.15, end=10.0, probability=0.9),
        TranscriptionWord(text="fly", start=10.0, end=11.0, probability=0.9),
        TranscriptionWord(text="To", start=13.72, end=14.20, probability=0.9),
    ]
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

    vocals_path = tmp_path / "vocals.wav"
    vocals_path.write_bytes(b"")

    adjusted, nudged = apply_forced_advisory_start_nudges(
        lines=lines,
        current_segments=current_segments,
        current_words=current_words,
        vocals_path=str(vocals_path),
        language="en",
        model_size="large",
        logger=_Logger(),
        load_aggressive_variant_fn=lambda **_kwargs: (
            aggressive_segments,
            aggressive_words,
            "en",
        ),
    )

    assert nudged == 1
    assert adjusted[1].start_time == 12.0
    assert adjusted[1].end_time == lines[1].end_time


def test_apply_forced_advisory_start_nudges_respects_disable_env(monkeypatch) -> None:
    monkeypatch.setenv("Y2K_DISABLE_FORCED_ADVISORY_START_NUDGE", "1")
    line = _line("I've been inclined", 12.74, 14.12)

    adjusted, nudged = apply_forced_advisory_start_nudges(
        lines=[line],
        current_segments=[],
        current_words=[],
        vocals_path="vocals.wav",
        language="en",
        model_size="large",
        logger=_Logger(),
        load_aggressive_variant_fn=lambda **_kwargs: ([], [], "en"),
    )

    assert nudged == 0
    assert adjusted[0].start_time == line.start_time


def test_apply_forced_advisory_start_nudges_skips_missing_audio(tmp_path) -> None:
    line = _line("I've been inclined", 12.74, 14.12)
    missing = tmp_path / "vocals.wav"

    adjusted, nudged = apply_forced_advisory_start_nudges(
        lines=[line],
        current_segments=[
            TranscriptionSegment(start=12.0, end=13.8, text="x", words=[])
        ],
        current_words=[],
        vocals_path=str(missing),
        language="en",
        model_size="large",
        logger=_Logger(),
        load_aggressive_variant_fn=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected advisory load")
        ),
    )

    assert nudged == 0
    assert adjusted[0].start_time == line.start_time
