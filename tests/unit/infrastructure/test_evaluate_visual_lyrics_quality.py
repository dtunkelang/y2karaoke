import importlib.util
import json
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "evaluate_visual_lyrics_quality.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_visual_lyrics_quality_module", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_split_tokens_with_optional_flags_marks_parenthetical_tokens():
    toks = _MODULE._split_tokens_with_optional_flags(
        "Come on (I'm in love with your body) now"
    )
    pairs = [(rec["token"], rec["optional"]) for rec in toks]
    assert ("come", False) in pairs
    assert ("on", False) in pairs
    assert ("i'm", True) in pairs
    assert ("body", True) in pairs
    assert ("now", False) in pairs


def test_summarize_alignment_reports_expected_counts():
    reference = ["a", "b", "c", "d"]
    extracted = ["a", "x", "c", "d", "e"]
    summary = _MODULE._summarize_alignment(reference, extracted, max_diff_blocks=4)
    assert summary["matched_token_count"] == 3
    assert abs(summary["precision"] - 0.6) < 1e-9
    assert abs(summary["recall"] - 0.75) < 1e-9
    assert summary["largest_diffs"]


def test_main_writes_output_json_and_uses_optional_parenthetical_mode(
    tmp_path, monkeypatch
):
    gold = {
        "lines": [
            {
                "words": [
                    {"text": "Come"},
                    {"text": "on"},
                    {"text": "now"},
                ]
            }
        ]
    }
    gold_path = tmp_path / "sample.gold.json"
    out_path = tmp_path / "report.json"
    gold_path.write_text(json.dumps(gold), encoding="utf-8")

    monkeypatch.setattr(
        _MODULE,
        "_load_lrc_tokens",
        lambda **_kwargs: ["come", "on", "im", "in", "love", "now"],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_visual_lyrics_quality.py",
            "--gold-json",
            str(gold_path),
            "--title",
            "Shape of You",
            "--artist",
            "Ed Sheeran",
            "--output-json",
            str(out_path),
        ],
    )
    code = _MODULE.main()
    assert code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["lrc_mode"] == "optional"
    assert payload["matched_token_count"] == 3
