import json

from y2karaoke.core.components.whisper.whisper_integration_forced_fallback import (
    attempt_whisperx_forced_alignment,
)
from y2karaoke.core.models import Line, Word


class _Logger:
    def debug(self, *_args, **_kwargs):
        return None

    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


def _line(start: float) -> Line:
    return Line(words=[Word(text="x", start_time=start, end_time=start + 0.4)])


def test_attempt_whisperx_forced_alignment_writes_stage_trace_when_enabled(
    tmp_path, monkeypatch
):
    lines = [_line(1.0), _line(2.0)]
    trace_path = tmp_path / "forced_trace.json"
    monkeypatch.setenv("Y2K_TRACE_FORCED_FALLBACK_STAGES_JSON", str(trace_path))
    monkeypatch.setenv("Y2K_TRACE_WHISPER_LINE_RANGE", "1-2")

    result = attempt_whisperx_forced_alignment(
        lines=lines,
        baseline_lines=lines,
        vocals_path="vocals.wav",
        language="en",
        detected_lang=None,
        logger=_Logger(),
        used_model="base",
        reason="test-trace",
        align_lines_with_whisperx_fn=lambda *_a, **_k: (
            lines,
            {
                "forced_word_coverage": 1.0,
                "forced_line_coverage": 1.0,
                "aligned_segments": [
                    {"start": 1.0, "end": 1.4, "text": "x"},
                    {"start": 2.0, "end": 2.4, "text": "x"},
                ],
            },
        ),
        should_rollback_short_line_degradation_fn=lambda *_a, **_k: (False, 0, 0),
        restore_implausibly_short_lines_fn=lambda *_a, **_k: (lines, 0),
        normalize_line_word_timings_fn=lambda current_lines: current_lines,
        enforce_monotonic_line_starts_fn=lambda current_lines: current_lines,
        enforce_non_overlapping_lines_fn=lambda current_lines: current_lines,
    )

    assert result is not None
    payload = json.loads(trace_path.read_text())
    assert payload["metadata"]["status"] == "accepted"
    assert payload["metadata"]["reason"] == "test-trace"
    assert payload["metadata"]["transcription_segment_count"] == 0
    assert payload["metadata"]["forced_segment_count"] == 2
    assert payload["metadata"]["forced_segment_preview"][0]["text"] == "x"
    stages = [snapshot["stage"] for snapshot in payload["snapshots"]]
    assert "baseline_lines" in stages
    assert "loaded_forced_alignment" in stages
    assert "after_finalize_forced_line_timing" in stages
    assert "final_forced_lines" in stages
