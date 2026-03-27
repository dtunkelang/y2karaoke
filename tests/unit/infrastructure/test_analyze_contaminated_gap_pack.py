import importlib.util
from pathlib import Path
import sys
from typing import Any, cast

_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "analyze_contaminated_gap_pack.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "analyze_contaminated_gap_pack_module", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
MODULE = cast(Any, _MODULE)


def test_analyze_contaminated_gap_pack_summarizes_effect_counts() -> None:
    report_doc = {
        "songs": [
            {
                "artist": "a-ha",
                "title": "Take On Me",
                "gold_path": "/tmp/take_gold.json",
                "report_path": "/tmp/take_report.json",
            },
            {
                "artist": "Bee Gees",
                "title": "Stayin' Alive",
                "gold_path": "/tmp/stay_gold.json",
                "report_path": "/tmp/stay_report.json",
            },
        ]
    }

    def fake_analyze(*, gold_json: Path, timing_report_json: Path) -> dict[str, object]:
        if "take_" in gold_json.name:
            return {
                "rows": [
                    {"effect": "prev_line_truncated"},
                    {"effect": "prev_line_truncated"},
                    {"effect": "mixed_or_small_effect"},
                ]
            }
        return {
            "rows": [
                {"effect": "prev_line_truncated"},
                {"effect": "next_line_delayed"},
            ]
        }

    fake_tool = cast(Any, type("GapTool", (), {"_analyze": staticmethod(fake_analyze)}))
    original = MODULE._load_gap_tool
    MODULE._load_gap_tool = lambda: fake_tool
    try:
        payload = MODULE.analyze(report_doc, match="Take On Me|Stayin")
    finally:
        MODULE._load_gap_tool = original

    assert payload["songs_analyzed"] == 2
    assert payload["gaps_total"] == 5
    assert payload["prev_line_truncated_total"] == 3
    assert payload["next_line_delayed_total"] == 1
    assert payload["rows"][0]["song"] == "a-ha - Take On Me"
    assert payload["rows"][0]["prev_line_truncated_count"] == 2
