from __future__ import annotations

import json
from pathlib import Path

from tools import analyze_pre_whisper_retained_lines as module


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_analyze_timing_report_detects_retained_upstream_miss(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.json"
    gold_path = tmp_path / "gold.json"
    _write_json(
        report_path,
        {
            "lines": [
                {
                    "index": 1,
                    "text": "You are",
                    "start": 0.81,
                    "end": 6.3,
                    "pre_whisper_start": 0.859,
                    "pre_whisper_end": 6.747,
                },
                {
                    "index": 2,
                    "text": "You are",
                    "start": 10.077,
                    "end": 15.966,
                    "pre_whisper_start": 10.156,
                    "pre_whisper_end": 16.045,
                },
            ]
        },
    )
    _write_json(
        gold_path,
        {
            "lines": [
                {"line_index": 1, "text": "You are", "start": 0.85, "end": 6.65},
                {"line_index": 2, "text": "You are", "start": 8.3, "end": 13.95},
            ]
        },
    )

    result = module.analyze_timing_report(
        report_path=report_path,
        gold_path=gold_path,
        max_retained_delta_sec=0.1,
    )

    assert result["candidate_count"] == 1
    row = result["rows"][0]
    assert row["line_index"] == 2
    assert row["gold_start_delta_sec"] == 1.777
    assert row["gold_end_delta_sec"] == 2.016


def test_analyze_timing_report_skips_lines_retimed_after_pre_whisper(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.json"
    gold_path = tmp_path / "gold.json"
    _write_json(
        report_path,
        {
            "lines": [
                {
                    "index": 1,
                    "text": "Late night when you need my love",
                    "start": 5.005,
                    "end": 7.69,
                    "pre_whisper_start": 5.005,
                    "pre_whisper_end": 7.17,
                }
            ]
        },
    )
    _write_json(
        gold_path,
        {
            "lines": [
                {
                    "line_index": 1,
                    "text": "Late night when you need my love",
                    "start": 4.95,
                    "end": 8.05,
                }
            ]
        },
    )

    result = module.analyze_timing_report(
        report_path=report_path,
        gold_path=gold_path,
        max_retained_delta_sec=0.1,
    )

    assert result["candidate_count"] == 0
