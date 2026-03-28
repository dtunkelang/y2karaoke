import json

from tools import analyze_shared_token_boundary_drift as tool


def test_analyze_shared_token_boundary_drift_reports_nearby_overlap_pair(tmp_path):
    timing = {
        "lines": [
            {
                "index": 1,
                "text": "You used to call me on my cell phone",
                "start": 0.76,
                "end": 3.18,
            },
            {
                "index": 2,
                "text": "Late night when you need my love",
                "start": 5.0,
                "end": 7.17,
            },
            {
                "index": 3,
                "text": "Call me on my cell phone",
                "start": 7.19,
                "end": 9.2,
            },
        ]
    }
    gold = {
        "lines": [
            {
                "line_index": 1,
                "text": "You used to call me on my cell phone",
                "end": 3.15,
                "start": 0.75,
            },
            {
                "line_index": 2,
                "text": "Late night when you need my love",
                "end": 8.05,
                "start": 4.95,
            },
            {
                "line_index": 3,
                "text": "Call me on my cell phone",
                "end": 10.35,
                "start": 8.55,
            },
        ]
    }
    timing_path = tmp_path / "timing.json"
    gold_path = tmp_path / "gold.json"
    timing_path.write_text(json.dumps(timing), encoding="utf-8")
    gold_path.write_text(json.dumps(gold), encoding="utf-8")

    result = tool.analyze(timing_path=timing_path, gold_path=gold_path)

    assert result["pair_count"] == 1
    row = result["rows"][0]
    assert row["left_index"] == 1
    assert row["right_index"] == 3
    assert row["line_gap"] == 2
    assert row["middle_index"] == 2
    assert row["middle_end_error_sec"] == -0.88
    assert row["right_start_error_sec"] == -1.36


def test_analyze_shared_token_boundary_drift_skips_low_overlap_pair(tmp_path):
    timing = {
        "lines": [
            {"index": 1, "text": "First line", "start": 0.0, "end": 1.0},
            {"index": 2, "text": "Totally different", "start": 1.1, "end": 2.0},
        ]
    }
    gold = {
        "lines": [
            {"line_index": 1, "text": "First line", "start": 0.0, "end": 1.0},
            {"line_index": 2, "text": "Totally different", "start": 1.1, "end": 2.0},
        ]
    }
    timing_path = tmp_path / "timing.json"
    gold_path = tmp_path / "gold.json"
    timing_path.write_text(json.dumps(timing), encoding="utf-8")
    gold_path.write_text(json.dumps(gold), encoding="utf-8")

    result = tool.analyze(timing_path=timing_path, gold_path=gold_path)

    assert result["pair_count"] == 0
