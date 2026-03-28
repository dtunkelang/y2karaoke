import json

from y2karaoke.core.components.alignment.timing_models import TranscriptionWord
from y2karaoke.core.components.whisper import whisper_mapping as wm
from y2karaoke.core.models import Line, Word


def test_extend_line_to_trailing_whisper_matches_writes_trace(
    tmp_path, monkeypatch
) -> None:
    trace_path = tmp_path / "tail_extension_trace.json"
    monkeypatch.setenv("Y2K_TRACE_TAIL_EXTENSION_JSON", str(trace_path))
    monkeypatch.setenv("Y2K_TRACE_WHISPER_LINE_RANGE", "1-1")

    lines = [
        Line(
            words=[
                Word(text="Take", start_time=4.22, end_time=5.48),
                Word(text="me", start_time=5.48, end_time=8.08),
                Word(text="on", start_time=8.08, end_time=8.22),
            ]
        ),
        Line(
            words=[
                Word(text="I'll", start_time=12.34, end_time=13.18),
                Word(text="be", start_time=13.18, end_time=13.84),
                Word(text="gone", start_time=13.84, end_time=15.5),
            ]
        ),
    ]
    whisper_words = [
        TranscriptionWord(start=4.28, end=5.64, text="take", probability=0.8),
        TranscriptionWord(start=5.64, end=8.16, text="me", probability=0.9),
        TranscriptionWord(start=8.16, end=9.64, text="on,", probability=0.7),
        TranscriptionWord(start=9.84, end=13.2, text="I'll", probability=0.9),
    ]

    wm._extend_line_to_trailing_whisper_matches(lines, whisper_words)

    payload = json.loads(trace_path.read_text())
    assert payload["lines"][0]["line_index"] == 1
    assert payload["lines"][0]["candidate_count"] == 1
    assert payload["lines"][0]["candidates"][0]["last_end"] == 9.64
