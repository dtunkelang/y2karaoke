from __future__ import annotations

import json
from pathlib import Path

from tools import analyze_pre_whisper_support_gaps as module


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_analyze_timing_report_detects_retained_support_gap(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    gold_path = tmp_path / "gold.json"
    _write_json(
        report_path,
        {
            "lines": [
                {
                    "index": 2,
                    "text": "You are",
                    "start": 10.08,
                    "end": 15.97,
                    "pre_whisper_start": 10.156,
                    "pre_whisper_end": 16.045,
                    "whisper_window_word_count": 1,
                    "whisper_window_words": [
                        {"text": "Confusion", "start": 28.64, "end": 30.04}
                    ],
                    "nearest_segment_start": 0.0,
                    "nearest_segment_end": 11.28,
                    "nearest_segment_end_start": 0.0,
                }
            ]
        },
    )
    _write_json(
        gold_path,
        {"lines": [{"line_index": 2, "text": "You are", "start": 8.3, "end": 13.95}]},
    )

    result = module.analyze_timing_report(report_path=report_path, gold_path=gold_path)

    assert result["candidate_count"] == 1
    row = result["rows"][0]
    assert row["line_index"] == 2
    assert row["window_overlap_ratio"] == 0.0
    assert row["nearest_segment_end"] == 11.28


def test_analyze_timing_report_skips_supported_boundary_case(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    gold_path = tmp_path / "gold.json"
    _write_json(
        report_path,
        {
            "lines": [
                {
                    "index": 3,
                    "text": "Call me on my cell phone",
                    "start": 7.74,
                    "end": 9.96,
                    "pre_whisper_start": 7.19,
                    "pre_whisper_end": 9.195,
                    "whisper_window_word_count": 8,
                    "whisper_window_words": [
                        {"text": "Call", "start": 7.74, "end": 8.0},
                        {"text": "me", "start": 8.0, "end": 8.2},
                        {"text": "on", "start": 8.2, "end": 8.35},
                        {"text": "my", "start": 8.35, "end": 8.5},
                        {"text": "cell", "start": 8.5, "end": 8.9},
                        {"text": "phone", "start": 8.9, "end": 9.2},
                    ],
                    "nearest_segment_start": 7.74,
                    "nearest_segment_end": 9.96,
                    "nearest_segment_end_start": 7.74,
                }
            ]
        },
    )
    _write_json(
        gold_path,
        {
            "lines": [
                {
                    "line_index": 3,
                    "text": "Call me on my cell phone",
                    "start": 8.55,
                    "end": 10.35,
                }
            ]
        },
    )

    result = module.analyze_timing_report(report_path=report_path, gold_path=gold_path)

    assert result["candidate_count"] == 0
