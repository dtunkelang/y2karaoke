from __future__ import annotations

import json
from pathlib import Path

from tools import analyze_forced_trace_transcription_support_gaps as module


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_analyze_trace_detects_loaded_line_without_transcription_support(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.json"
    _write_json(
        trace_path,
        {
            "metadata": {
                "transcription_segment_count": 1,
                "transcription_segment_preview": [
                    {
                        "start": 28.71,
                        "end": 31.33,
                        "text": "Confusion never stops",
                    }
                ],
            },
            "snapshots": [
                {
                    "stage": "loaded_forced_alignment",
                    "lines": [
                        {
                            "line_index": 2,
                            "text": "You are",
                            "start": 9.827,
                            "end": 10.549,
                            "duration": 0.722,
                        },
                        {
                            "line_index": 3,
                            "text": "Confusion that never stops",
                            "start": 29.116,
                            "end": 32.0,
                            "duration": 2.884,
                        },
                    ],
                },
                {
                    "stage": "after_sustained_line_repair",
                    "lines": [
                        {
                            "line_index": 2,
                            "text": "You are",
                            "start": 9.827,
                            "end": 15.715,
                            "duration": 5.888,
                        },
                        {
                            "line_index": 3,
                            "text": "Confusion that never stops",
                            "start": 29.116,
                            "end": 32.0,
                            "duration": 2.884,
                        },
                    ],
                },
            ],
        },
    )

    result = module.analyze_trace(trace_path=trace_path)

    assert result["candidate_count"] == 1
    row = result["rows"][0]
    assert row["line_index"] == 2
    assert row["best_transcription_overlap_ratio"] == 0.0
    assert row["best_transcription_segment"] is None
    assert row["sustained_end"] == 15.715
    assert row["sustained_gain_sec"] == 5.166


def test_analyze_trace_skips_line_with_supported_transcription(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json"
    _write_json(
        trace_path,
        {
            "metadata": {
                "transcription_segment_count": 3,
                "transcription_segment_preview": [
                    {
                        "start": 0.4,
                        "end": 2.84,
                        "text": "You used to call me on my cell phone",
                    },
                    {
                        "start": 2.84,
                        "end": 7.74,
                        "text": "Late night when you need my love",
                    },
                    {"start": 7.74, "end": 9.96, "text": "Call me on my cell phone"},
                ],
            },
            "snapshots": [
                {
                    "stage": "loaded_forced_alignment",
                    "lines": [
                        {
                            "line_index": 2,
                            "text": "Late night when you need my love",
                            "start": 5.005,
                            "end": 7.17,
                            "duration": 2.165,
                        }
                    ],
                },
                {
                    "stage": "after_sustained_line_repair",
                    "lines": [
                        {
                            "line_index": 2,
                            "text": "Late night when you need my love",
                            "start": 5.005,
                            "end": 7.17,
                            "duration": 2.165,
                        }
                    ],
                },
            ],
        },
    )

    result = module.analyze_trace(trace_path=trace_path)

    assert result["candidate_count"] == 0
