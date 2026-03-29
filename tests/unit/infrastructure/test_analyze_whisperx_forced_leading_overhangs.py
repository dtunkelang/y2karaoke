from __future__ import annotations

import json
from pathlib import Path

from tools import analyze_whisperx_forced_leading_overhangs as module


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_analyze_trace_detects_low_confidence_leading_if_overhang(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.json"
    _write_json(
        trace_path,
        {
            "aligned_segments": [
                {
                    "text": "If you ain't runnin' game",
                    "words": [
                        {"word": "If", "start": 5.539, "end": 6.022, "score": 0.346},
                        {"word": "you", "start": 6.062, "end": 6.223, "score": 0.992},
                    ],
                }
            ],
            "line_mappings": [
                {
                    "line_index": 0,
                    "line_text": "If you ain't runnin' game",
                    "segment_start": 5.539,
                    "segment_end": 7.773,
                    "segment_words": [
                        {"text": "If", "start": 5.539, "end": 6.022},
                        {"text": "you", "start": 6.062, "end": 6.223},
                        {"text": "ain't", "start": 6.243, "end": 6.726},
                        {"text": "runnin'", "start": 6.787, "end": 7.37},
                        {"text": "game", "start": 7.431, "end": 7.773},
                    ],
                }
            ],
        },
    )

    result = module.analyze_trace(trace_path=trace_path)

    assert result["candidate_count"] == 1
    row = result["rows"][0]
    assert row["line_index"] == 1
    assert row["first_word"] == "If"
    assert row["first_score"] == 0.346
    assert row["leading_overhang_sec"] == 0.483


def test_analyze_trace_skips_non_light_or_higher_confidence_overhangs(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.json"
    _write_json(
        trace_path,
        {
            "aligned_segments": [
                {
                    "text": "Call me on my cell phone",
                    "words": [
                        {"word": "Call", "start": 7.19, "end": 7.451, "score": 0.373},
                    ],
                },
                {
                    "text": "You are",
                    "words": [
                        {"word": "You", "start": 0.81, "end": 3.76, "score": 0.93},
                    ],
                },
            ],
            "line_mappings": [
                {
                    "line_index": 0,
                    "line_text": "Call me on my cell phone",
                    "segment_start": 7.19,
                    "segment_end": 9.195,
                    "segment_words": [
                        {"text": "Call", "start": 7.19, "end": 7.451},
                        {"text": "me", "start": 7.491, "end": 7.651},
                        {"text": "on", "start": 7.852, "end": 8.012},
                        {"text": "my", "start": 8.032, "end": 8.092},
                        {"text": "cell", "start": 8.534, "end": 8.955},
                        {"text": "phone", "start": 8.975, "end": 9.195},
                    ],
                },
                {
                    "line_index": 1,
                    "line_text": "You are",
                    "segment_start": 0.81,
                    "segment_end": 6.289,
                    "segment_words": [
                        {"text": "You", "start": 0.81, "end": 3.76},
                        {"text": "are", "start": 3.94, "end": 6.289},
                    ],
                },
            ],
        },
    )

    result = module.analyze_trace(trace_path=trace_path)

    assert result["candidate_count"] == 0
