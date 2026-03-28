from __future__ import annotations

import json
from pathlib import Path

from tools import analyze_forced_trace_exact_segment_boundaries as module


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_analyze_trace_detects_pre_repair_hotline_boundary(tmp_path: Path) -> None:
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
                    {
                        "start": 7.74,
                        "end": 9.96,
                        "text": "Call me on my cell phone",
                    },
                ],
            },
            "snapshots": [
                {
                    "stage": "after_post_finalize_refrain_repairs",
                    "lines": [
                        {
                            "line_index": 1,
                            "start": 0.757,
                            "end": 3.184,
                            "text": "You used to call me on my cell phone",
                        },
                        {
                            "line_index": 2,
                            "start": 5.005,
                            "end": 7.17,
                            "text": "Late night when you need my love",
                        },
                        {
                            "line_index": 3,
                            "start": 7.19,
                            "end": 9.195,
                            "text": "Call me on my cell phone",
                        },
                    ],
                },
                {
                    "stage": "after_restore_exact_segment_boundaries",
                    "lines": [
                        {
                            "line_index": 1,
                            "start": 0.757,
                            "end": 3.184,
                            "text": "You used to call me on my cell phone",
                        },
                        {
                            "line_index": 2,
                            "start": 5.005,
                            "end": 7.69,
                            "text": "Late night when you need my love",
                        },
                        {
                            "line_index": 3,
                            "start": 7.74,
                            "end": 9.96,
                            "text": "Call me on my cell phone",
                        },
                    ],
                },
            ],
        },
    )

    result = module.analyze_trace(trace_path=trace_path)

    assert result["candidate_count"] == 1
    row = result["rows"][0]
    assert row["stage"] == "after_post_finalize_refrain_repairs"
    assert row["left_index"] == 2
    assert row["right_index"] == 3
    assert row["tail_shortfall_sec"] == 0.57
    assert row["next_early_start_sec"] == 0.55


def test_analyze_trace_skips_resolved_boundary(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json"
    _write_json(
        trace_path,
        {
            "metadata": {
                "transcription_segment_count": 2,
                "transcription_segment_preview": [
                    {
                        "start": 1.011,
                        "end": 3.656,
                        "text": "Please please please",
                    },
                    {
                        "start": 4.392,
                        "end": 6.442,
                        "text": "Don't prove I'm right",
                    },
                ],
            },
            "snapshots": [
                {
                    "stage": "final_forced_lines",
                    "lines": [
                        {
                            "line_index": 1,
                            "start": 1.011,
                            "end": 3.656,
                            "text": "Please please please",
                        },
                        {
                            "line_index": 2,
                            "start": 4.392,
                            "end": 6.442,
                            "text": "Don't prove I'm right",
                        },
                    ],
                }
            ],
        },
    )

    result = module.analyze_trace(trace_path=trace_path)

    assert result["candidate_count"] == 0
