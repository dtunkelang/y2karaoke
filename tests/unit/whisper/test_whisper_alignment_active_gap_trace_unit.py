import json

import numpy as np

import y2karaoke.core.components.whisper.whisper_alignment_refinement as war
from y2karaoke.core.components.alignment.timing_models import AudioFeatures
from y2karaoke.core.models import Line, Word


def test_extend_line_ends_across_active_gaps_writes_trace(
    tmp_path, monkeypatch
) -> None:
    trace_path = tmp_path / "active_gap_trace.json"
    monkeypatch.setenv("Y2K_TRACE_ACTIVE_GAP_EXTENSION_JSON", str(trace_path))
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
        count = war._extend_line_ends_across_active_gaps(lines, audio_features)

    assert count == 1
    payload = json.loads(trace_path.read_text())
    assert payload["rows"][0]["decision"] == "extend"
    assert payload["rows"][0]["new_end"] == 13.45
