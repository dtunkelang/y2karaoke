import importlib.util
from pathlib import Path
import sys

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "analyze_two_layer_guard_tradeoffs.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "analyze_two_layer_guard_tradeoffs_module", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_analyze_two_layer_guard_tradeoffs_prefers_isolated_candidate() -> None:
    report_doc = {
        "songs": [
            {"artist": "Daddy Yankee & Snow", "title": "Con Calma"},
            {"artist": "a-ha", "title": "Take On Me"},
        ]
    }

    def fake_analyze(
        report_doc: dict[str, object],
        *,
        min_text_similarity: float = 0.58,
        min_token_overlap: float = 0.5,
        min_line_words: int = 6,
        min_anchor_surplus_words: int = 15,
        min_anchor_words: int = 20,
    ) -> dict[str, object]:
        assert report_doc["songs"]
        assert min_text_similarity == 0.58
        assert min_token_overlap == 0.5
        assert min_line_words == 6
        assert min_anchor_surplus_words == 15
        if min_anchor_words == 15:
            return {
                "baseline_coverage_ratio_total": 0.4643,
                "adjusted_coverage_ratio_total": 0.6562,
                "rows": [
                    {
                        "song": "Daddy Yankee & Snow - Con Calma",
                        "recovered_lines": 8,
                    },
                    {
                        "song": "a-ha - Take On Me",
                        "recovered_lines": 1,
                    },
                ],
            }
        return {
            "baseline_coverage_ratio_total": 0.4643,
            "adjusted_coverage_ratio_total": 0.6562,
            "rows": [
                {
                    "song": "Daddy Yankee & Snow - Con Calma",
                    "recovered_lines": 8,
                },
                {
                    "song": "a-ha - Take On Me",
                    "recovered_lines": 0,
                },
            ],
        }

    original = _MODULE.pack_tool.analyze
    _MODULE.pack_tool.analyze = fake_analyze
    try:
        payload = _MODULE.analyze(
            report_doc,
            guard_candidates=[(6, 15, 15), (6, 15, 20)],
        )
    finally:
        _MODULE.pack_tool.analyze = original

    best = payload["best_candidate"]
    assert best["min_anchor_words"] == 20
    assert best["spillover_song_count"] == 0
