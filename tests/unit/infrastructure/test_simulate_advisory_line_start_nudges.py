from tools import simulate_advisory_line_start_nudges as tool


def test_simulate_song_applies_candidate_start_only(monkeypatch) -> None:
    song = {
        "artist": "Neil Diamond",
        "title": "Sweet Caroline",
        "report_path": "/tmp/report.json",
        "gold_path": "/tmp/gold.json",
    }
    candidate_lines = {
        1: {
            "advisory_start": 12.0,
            "aggressive_best_segment_text": "I've been inclined",
        }
    }

    report = {
        "lines": [
            {"index": 1, "text": "I've been inclined", "start": 12.74, "end": 14.12}
        ]
    }
    gold = {"lines": [{"start": 11.95, "end": 14.6}]}

    monkeypatch.setattr(tool.support_tool, "_load_report", lambda path: report)
    monkeypatch.setattr(tool, "_load_gold", lambda path: gold)
    result = tool._simulate_song(song=song, candidate_lines=candidate_lines)

    assert result["simulated_start_mean"] < result["current_start_mean"]
    assert result["lines"][0]["simulated_start"] == 12.0
