from __future__ import annotations

import json
from pathlib import Path

from tools import analyze_forced_trace_sustained_repairs as module


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_analyze_trace_detects_large_sustained_expansion(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json"
    _write_json(
        trace_path,
        {
            "snapshots": [
                {
                    "stage": "loaded_forced_alignment",
                    "lines": [
                        {
                            "line_index": 1,
                            "start": 0.81,
                            "end": 6.304,
                            "duration": 5.494,
                            "text": "You are",
                        },
                        {
                            "line_index": 2,
                            "start": 9.827,
                            "end": 10.549,
                            "duration": 0.722,
                            "text": "You are",
                        },
                    ],
                },
                {
                    "stage": "after_sustained_line_repair",
                    "lines": [
                        {
                            "line_index": 1,
                            "start": 0.81,
                            "end": 6.304,
                            "duration": 5.494,
                            "text": "You are",
                        },
                        {
                            "line_index": 2,
                            "start": 9.827,
                            "end": 15.716,
                            "duration": 5.889,
                            "text": "You are",
                        },
                    ],
                },
                {
                    "stage": "after_finalize_forced_line_timing",
                    "lines": [
                        {
                            "line_index": 2,
                            "start": 10.077,
                            "end": 15.966,
                            "duration": 5.889,
                            "text": "You are",
                        }
                    ],
                },
            ]
        },
    )

    result = module.analyze_trace(trace_path=trace_path)

    assert result["candidate_count"] == 1
    row = result["rows"][0]
    assert row["line_index"] == 2
    assert row["duration_gain_sec"] == 5.167
    assert row["duration_gain_ratio"] == 8.157
    assert row["final_start"] == 10.077
    assert row["final_end"] == 15.966


def test_analyze_trace_skips_small_sustained_change(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json"
    _write_json(
        trace_path,
        {
            "snapshots": [
                {
                    "stage": "loaded_forced_alignment",
                    "lines": [
                        {
                            "line_index": 1,
                            "start": 5.005,
                            "end": 7.17,
                            "duration": 2.165,
                            "text": "Late night when you need my love",
                        }
                    ],
                },
                {
                    "stage": "after_sustained_line_repair",
                    "lines": [
                        {
                            "line_index": 1,
                            "start": 5.005,
                            "end": 7.17,
                            "duration": 2.165,
                            "text": "Late night when you need my love",
                        }
                    ],
                },
            ]
        },
    )

    result = module.analyze_trace(trace_path=trace_path)

    assert result["candidate_count"] == 0
