import json

from y2karaoke.core.components.alignment.timing_models import (
    TranscriptionSegment,
    TranscriptionWord,
)
from y2karaoke.core.components.whisper.whisper_forced_advisory_trace import (
    _aggressive_cache_is_usable,
    _resolve_advisory_audio_path,
    build_forced_advisory_trace_payload,
    maybe_write_forced_advisory_trace,
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


def test_build_forced_advisory_trace_payload_ranks_high_confidence_candidate() -> None:
    lines = [_line("I've been inclined", 12.74, 14.12)]
    current_segments = [
        TranscriptionSegment(start=8.72, end=25.18, text="The night", words=[])
    ]
    current_words: list[TranscriptionWord] = []
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

    payload = build_forced_advisory_trace_payload(
        lines=lines,
        current_segments=current_segments,
        current_words=current_words,
        aggressive_segments=aggressive_segments,
        aggressive_words=aggressive_words,
    )

    assert len(payload["candidates"]) == 1
    assert payload["candidates"][0]["bucket"] == "high_confidence"
    assert payload["candidates"][0]["text"] == "I've been inclined"


def test_aggressive_cache_is_usable_rejects_single_merged_segment() -> None:
    words = [
        TranscriptionWord(
            text=f"w{idx}",
            start=float(idx),
            end=float(idx + 1),
            probability=0.9,
        )
        for idx in range(9)
    ]
    segments = [
        TranscriptionSegment(start=0.0, end=9.0, text="merged text", words=words)
    ]

    assert _aggressive_cache_is_usable(segments, words) is False


def test_resolve_advisory_audio_path_prefers_companion_clip(tmp_path) -> None:
    vocals = tmp_path / "clip_(Vocals)_htdemucs_ft.wav"
    clip = tmp_path / "clip.wav"
    vocals.write_bytes(b"")
    clip.write_bytes(b"")

    assert _resolve_advisory_audio_path(str(vocals)) == str(clip)


def test_maybe_write_forced_advisory_trace_writes_payload(
    tmp_path, monkeypatch
) -> None:
    trace_path = tmp_path / "advisory.json"
    vocals_path = tmp_path / "vocals.wav"
    vocals_path.write_bytes(b"")
    monkeypatch.setenv("Y2K_TRACE_FORCED_ADVISORY_JSON", str(trace_path))
    current_segments = [
        TranscriptionSegment(start=8.72, end=25.18, text="The night", words=[])
    ]
    current_words: list[TranscriptionWord] = []
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

    payload = maybe_write_forced_advisory_trace(
        lines=[_line("I've been inclined", 12.74, 14.12)],
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

    assert payload is not None
    saved = json.loads(trace_path.read_text())
    assert saved["candidates"][0]["bucket"] == "high_confidence"
    assert saved["aggressive_language"] == "en"
