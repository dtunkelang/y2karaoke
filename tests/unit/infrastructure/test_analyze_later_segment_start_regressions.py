from __future__ import annotations

import json
from pathlib import Path

from tools import analyze_later_segment_start_regressions as module


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_analyze_timing_report_detects_later_segment_start_regression(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.json"
    gold_path = tmp_path / "gold.json"
    _write_json(
        report_path,
        {
            "lines": [
                {
                    "index": 3,
                    "text": "If you ain't runnin' game",
                    "start": 5.539,
                    "pre_whisper_start": 5.889,
                    "nearest_segment_start": 6.14,
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
                    "text": "If you ain't runnin' game",
                    "start": 6.35,
                    "end": 7.8,
                }
            ]
        },
    )

    result = module.analyze_timing_report(report_path=report_path, gold_path=gold_path)

    assert result["candidate_count"] == 1
    row = result["rows"][0]
    assert row["pre_whisper_regression_sec"] == 0.35
    assert row["segment_gain_sec"] == 0.601
    assert row["gold_gain_sec"] == 0.811


def test_analyze_timing_report_skips_lines_without_real_later_support(
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
                    "text": "Say my name, say my name",
                    "start": 0.86,
                    "pre_whisper_start": 0.906,
                    "nearest_segment_start": 0.0,
                }
            ]
        },
    )
    _write_json(
        gold_path,
        {
            "lines": [
                {"line_index": 1, "text": "Say my name, say my name", "start": 0.85}
            ]
        },
    )

    result = module.analyze_timing_report(report_path=report_path, gold_path=gold_path)

    assert result["candidate_count"] == 0
