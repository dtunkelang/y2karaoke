import json

from tools import analyze_alternating_hook_family as tool


def test_analyze_alternating_hook_family_finds_take_on_me_shape(tmp_path):
    gold = {
        "lines": [
            {"text": "Take on me"},
            {"text": "Take me on"},
            {"text": "I'll be gone"},
        ]
    }
    (tmp_path / "sample.gold.json").write_text(json.dumps(gold), encoding="utf-8")

    result = tool.analyze(gold_root=tmp_path)

    assert result["match_count"] == 1
    assert result["rows"][0]["line_index"] == 1
    assert result["rows"][0]["first_text"] == "Take on me"
    assert result["rows"][0]["second_text"] == "Take me on"


def test_analyze_alternating_hook_family_skips_non_alternating_pairs(tmp_path):
    gold = {
        "lines": [
            {"text": "Take on me"},
            {"text": "Take on me"},
            {"text": "I'll be gone"},
        ]
    }
    (tmp_path / "sample.gold.json").write_text(json.dumps(gold), encoding="utf-8")

    result = tool.analyze(gold_root=tmp_path)

    assert result["match_count"] == 0


def test_analyze_alternating_hook_family_recurses(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    gold = {
        "lines": [
            {"text": "Take on me"},
            {"text": "Take me on"},
        ]
    }
    (nested / "sample.gold.json").write_text(json.dumps(gold), encoding="utf-8")

    result = tool.analyze(gold_root=tmp_path)

    assert result["match_count"] == 1
