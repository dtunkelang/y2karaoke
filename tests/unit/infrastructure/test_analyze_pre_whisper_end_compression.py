from __future__ import annotations

import json
from pathlib import Path

from tools import analyze_pre_whisper_end_compression as module


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_analyze_timing_report_detects_end_compression_regression(
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
                    "end": 6.29,
                    "pre_whisper_start": 0.859,
                    "pre_whisper_end": 6.512,
                }
            ]
        },
    )
    _write_json(
        gold_path,
        {"lines": [{"line_index": 1, "text": "You are", "start": 0.85, "end": 6.65}]},
    )

    result = module.analyze_timing_report(report_path=report_path, gold_path=gold_path)

    assert result["candidate_count"] == 1
    row = result["rows"][0]
    assert row["line_index"] == 1
    assert row["end_compression_sec"] == 0.222
    assert row["gold_end_regression_sec"] == 0.222


def test_analyze_timing_report_skips_end_compression_without_gold_regression(
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
                    "text": "Hook line",
                    "start": 2.0,
                    "end": 4.8,
                    "pre_whisper_start": 2.0,
                    "pre_whisper_end": 5.1,
                }
            ]
        },
    )
    _write_json(
        gold_path,
        {"lines": [{"line_index": 1, "text": "Hook line", "start": 2.0, "end": 4.75}]},
    )

    result = module.analyze_timing_report(report_path=report_path, gold_path=gold_path)

    assert result["candidate_count"] == 0
