import json

from y2karaoke.core.components.whisper.whisper_integration_baseline import (
    _constrain_line_starts_to_baseline,
)
from y2karaoke.core.models import Line, Word


def test_constrain_line_starts_to_baseline_writes_trace(tmp_path, monkeypatch) -> None:
    trace_path = tmp_path / "baseline_constraint_trace.json"
    monkeypatch.setenv("Y2K_TRACE_BASELINE_CONSTRAINT_JSON", str(trace_path))

    mapped = [
        Line(
            words=[
                Word(text="Take", start_time=4.22, end_time=5.48),
                Word(text="me", start_time=5.48, end_time=8.08),
                Word(text="on", start_time=8.08, end_time=9.72),
            ]
        ),
        Line(
            words=[
                Word(text="I'll", start_time=12.34, end_time=13.18),
                Word(text="be", start_time=13.18, end_time=13.84),
                Word(text="gone", start_time=13.84, end_time=15.75),
            ]
        ),
    ]
    baseline = [
        Line(
            words=[
                Word(text="Take", start_time=6.451, end_time=7.2),
                Word(text="me", start_time=7.2, end_time=8.0),
                Word(text="on", start_time=8.0, end_time=8.8),
            ]
        ),
        Line(
            words=[
                Word(text="I'll", start_time=11.912, end_time=12.6),
                Word(text="be", start_time=12.6, end_time=13.2),
                Word(text="gone", start_time=13.2, end_time=14.8),
            ]
        ),
    ]

    constrained = _constrain_line_starts_to_baseline(mapped, baseline)

    assert constrained[0].start_time == 6.451
    payload = json.loads(trace_path.read_text())
    assert payload["rows"][0]["decision"] == "shift_to_baseline"
    assert payload["rows"][0]["shift"] == 2.231
    assert payload["rows"][0]["compressed_to_next_baseline"] == 11.912
