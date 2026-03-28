from __future__ import annotations

import json
from pathlib import Path

from tools import analyze_aggregate_segment_merges as module


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_analyze_aggregate_segment_merges_detects_hotline_shape(tmp_path: Path) -> None:
    aggregate_path = tmp_path / "aggregate.json"
    vocals_path = tmp_path / "vocals.json"
    gold_path = tmp_path / "gold.json"

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
                {
                    "start": 7.74,
                    "end": 9.96,
                    "text": "Call me on my cell phone",
                },
            ]
        },
    )
    _write_json(
        gold_path,
        {
            "lines": [
                {"line_index": 1, "text": "You used to call me on my cell phone"},
                {"line_index": 2, "text": "Late night when you need my love"},
                {"line_index": 3, "text": "Call me on my cell phone"},
            ]
        },
    )

    result = module.analyze(
        aggregate_path=aggregate_path,
        vocals_path=vocals_path,
        gold_path=gold_path,
    )

    assert result["merge_count"] == 1
    assert result["rows"] == [
        {
            "left_line_index": 1,
            "right_line_index": 2,
            "left_text": "You used to call me on my cell phone",
            "right_text": "Late night when you need my love",
            "aggregate_segment_index": 1,
            "aggregate_start": 0.0,
            "aggregate_end": 7.76,
            "aggregate_text": (
                "You used to call me on my cell phone "
                "Late night when you need my love"
            ),
            "vocals_left_segment_index": 1,
            "vocals_right_segment_index": 2,
            "vocals_left_start": 0.4,
            "vocals_left_end": 2.84,
            "vocals_right_start": 2.84,
            "vocals_right_end": 7.74,
        }
    ]


def test_analyze_aggregate_segment_merges_skips_when_both_paths_split(
    tmp_path: Path,
) -> None:
    aggregate_path = tmp_path / "aggregate.json"
    vocals_path = tmp_path / "vocals.json"
    gold_path = tmp_path / "gold.json"

    payload = {
        "segments": [
            {"start": 0.0, "end": 2.4, "text": "Say my name, say my name"},
            {"start": 2.4, "end": 4.48, "text": "If no one is around you"},
            {"start": 4.48, "end": 6.14, "text": "Say baby I love you"},
            {"start": 6.14, "end": 7.56, "text": "If you ain't one in game"},
        ]
    }
    _write_json(aggregate_path, payload)
    _write_json(vocals_path, payload)
    _write_json(
        gold_path,
        {
            "lines": [
                {"line_index": 1, "text": "Say my name, say my name"},
                {
                    "line_index": 2,
                    "text": 'If no one is around you, say, "Baby, I love you"',
                },
                {"line_index": 3, "text": "If you ain't runnin' game"},
            ]
        },
    )

    result = module.analyze(
        aggregate_path=aggregate_path,
        vocals_path=vocals_path,
        gold_path=gold_path,
    )

    assert result["merge_count"] == 0
    assert result["rows"] == []
