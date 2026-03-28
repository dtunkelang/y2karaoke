from __future__ import annotations

import json
from pathlib import Path

from tools import analyze_merge_reuse_pressure as module


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_analyze_merge_reuse_pressure_detects_hotline_shape(tmp_path: Path) -> None:
    timing_path = tmp_path / "timing.json"
    gold_path = tmp_path / "gold.json"
    aggregate_path = tmp_path / "aggregate.json"
    vocals_path = tmp_path / "vocals.json"

    _write_json(
        timing_path,
        {
            "lines": [
                {
                    "index": 1,
                    "start": 0.76,
                    "end": 3.18,
                    "text": "You used to call me on my cell phone",
                },
                {
                    "index": 2,
                    "start": 5.0,
                    "end": 7.17,
                    "text": "Late night when you need my love",
                },
                {
                    "index": 3,
                    "start": 7.19,
                    "end": 9.2,
                    "text": "Call me on my cell phone",
                },
            ]
        },
    )
    _write_json(
        gold_path,
        {
            "lines": [
                {
                    "line_index": 1,
                    "start": 0.75,
                    "end": 3.15,
                    "text": "You used to call me on my cell phone",
                },
                {
                    "line_index": 2,
                    "start": 4.95,
                    "end": 8.05,
                    "text": "Late night when you need my love",
                },
                {
                    "line_index": 3,
                    "start": 8.55,
                    "end": 10.35,
                    "text": "Call me on my cell phone",
                },
            ]
        },
    )
    _write_json(
        aggregate_path,
        {
            "segments": [
                {
                    "start": 0.0,
                    "end": 7.76,
                    "text": (
                        "You used to call me on my cell phone "
                        "Late night when you need my love"
                    ),
                },
                {
                    "start": 7.76,
                    "end": 9.96,
                    "text": "Call me on my cell phone",
                },
            ]
        },
    )
    _write_json(
        vocals_path,
        {
            "segments": [
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
            ]
        },
    )

    result = module.analyze(
        timing_path=timing_path,
        gold_path=gold_path,
        aggregate_path=aggregate_path,
        vocals_path=vocals_path,
    )

    assert result["triplet_count"] == 1
    row = result["rows"][0]
    assert row["reuse_overlap_ratio"] == 0.667
    assert row["aggregate_segment_gap_sec"] == 0.0
    assert row["middle_end_error_sec"] == -0.88
    assert row["right_start_error_sec"] == -1.36
    assert row["vocals_split_ok"] is True


def test_analyze_merge_reuse_pressure_distinguishes_healthy_control(
    tmp_path: Path,
) -> None:
    timing_path = tmp_path / "timing.json"
    gold_path = tmp_path / "gold.json"
    aggregate_path = tmp_path / "aggregate.json"
    vocals_path = tmp_path / "vocals.json"

    _write_json(
        timing_path,
        {
            "lines": [
                {
                    "index": 1,
                    "start": 1.011,
                    "end": 3.656,
                    "text": "Please, please, please",
                },
                {
                    "index": 2,
                    "start": 4.392,
                    "end": 6.442,
                    "text": "Don't prove I'm right",
                },
                {
                    "index": 3,
                    "start": 9.782,
                    "end": 12.736,
                    "text": "And please, please, please",
                },
            ]
        },
    )
    _write_json(
        gold_path,
        {
            "lines": [
                {
                    "line_index": 1,
                    "start": 1.011,
                    "end": 3.656,
                    "text": "Please, please, please",
                },
                {
                    "line_index": 2,
                    "start": 4.392,
                    "end": 6.442,
                    "text": "Don't prove I'm right",
                },
                {
                    "line_index": 3,
                    "start": 9.782,
                    "end": 12.736,
                    "text": "And please, please, please",
                },
            ]
        },
    )
    _write_json(
        aggregate_path,
        {
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.68,
                    "text": "Please, please, please don't prove I'm right",
                },
                {
                    "start": 9.38,
                    "end": 18.04,
                    "text": (
                        "Please, please, please don't bring me to tears "
                        "when I just did my makeup so nice"
                    ),
                },
            ]
        },
    )
    _write_json(
        vocals_path,
        {
            "segments": [
                {"start": 0.0, "end": 3.28, "text": "Please, please, please"},
                {"start": 3.84, "end": 5.44, "text": "Don't prove I'm right"},
                {"start": 9.38, "end": 12.18, "text": "Please, please, please"},
            ]
        },
    )

    result = module.analyze(
        timing_path=timing_path,
        gold_path=gold_path,
        aggregate_path=aggregate_path,
        vocals_path=vocals_path,
    )

    assert result["triplet_count"] == 1
    row = result["rows"][0]
    assert row["reuse_overlap_ratio"] == 0.5
    assert row["aggregate_segment_gap_sec"] == 3.7
    assert row["middle_end_error_sec"] == 0.0
    assert row["right_start_error_sec"] == 0.0
    assert row["vocals_split_ok"] is True
