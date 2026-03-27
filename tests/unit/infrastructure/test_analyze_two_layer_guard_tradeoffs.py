import importlib.util
import json
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


def test_analyze_two_layer_guard_tradeoffs_prefers_isolated_candidate(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "benchmark_report.json"
    report_path.write_text(
        json.dumps(
            {
                "songs": [
                    {
                        "artist": "Daddy Yankee & Snow",
                        "title": "Con Calma",
                        "report_path": str(
                            Path(
                                "benchmarks/results/20260327T193726Z/"
                                "01_daddy-yankee-snow-con-calma-first-chorus-bilingual_"
                                "timing_report.json"
                            ).resolve()
                        ),
                    },
                    {
                        "artist": "a-ha",
                        "title": "Take On Me",
                        "report_path": str(
                            Path(
                                "benchmarks/results/20260327T180340Z/"
                                "03_a-ha-take-on-me-first-chorus_timing_report.json"
                            ).resolve()
                        ),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = _MODULE.analyze(
        json.loads(report_path.read_text()),
        guard_candidates=[(6, 15, 15), (6, 15, 20)],
    )

    best = payload["best_candidate"]
    assert best["min_anchor_words"] == 20
    assert best["spillover_song_count"] == 0
