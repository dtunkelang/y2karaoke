import json

import numpy as np

import y2karaoke.core.components.whisper.whisper_alignment_refinement as war
from y2karaoke.core.components.alignment.timing_models import AudioFeatures
from y2karaoke.core.models import Line, Word


def test_pull_lines_forward_for_continuous_vocals_writes_stage_trace(
    tmp_path, monkeypatch
) -> None:
    trace_path = tmp_path / "continuous_vocals_trace.json"
    monkeypatch.setenv("Y2K_TRACE_CONTINUOUS_VOCALS_JSON", str(trace_path))
    monkeypatch.setenv("Y2K_TRACE_WHISPER_LINE_RANGE", "1-2")
    lines = [
        Line(words=[Word(text="one", start_time=10.0, end_time=11.0)]),
        Line(words=[Word(text="two", start_time=13.5, end_time=14.0)]),
    ]
    audio_features = AudioFeatures(
        onset_times=np.array([10.5, 12.0, 13.0], dtype=float),
        silence_regions=[],
        vocal_start=0.0,
        vocal_end=20.0,
        duration=20.0,
        energy_envelope=np.array([], dtype=float),
        energy_times=np.array([], dtype=float),
    )

    with war.use_alignment_refinement_hooks(
        check_vocal_activity_in_range_fn=lambda *args: 0.9,
        check_for_silence_in_range_fn=lambda *args, **kw: False,
    ):
        war._pull_lines_forward_for_continuous_vocals(lines, audio_features)

    payload = json.loads(trace_path.read_text())
    call_index = payload["rows"][0]["call_index"]
    assert isinstance(call_index, int)
    assert call_index >= 1
    assert payload["rows"][0]["stage"] == "before"
    assert payload["rows"][1]["call_index"] == call_index
    assert payload["rows"][1]["stage"] == "after_shift_long_activity_gaps"
    assert payload["rows"][2]["call_index"] == call_index
    assert payload["rows"][2]["stage"] == "after_extend_active_gaps"
