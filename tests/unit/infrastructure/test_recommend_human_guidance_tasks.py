import importlib.util
import json
from pathlib import Path
import sys

_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "recommend_human_guidance_tasks.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "recommend_human_guidance_tasks_module", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_recommend_human_guidance_tasks_prioritizes_problem_song(
    tmp_path, monkeypatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report = {
        "songs": [
            {
                "artist": "A",
                "title": "Good",
                "status": "ok",
                "metrics": {
                    "agreement_coverage_ratio": 0.8,
                    "agreement_start_p95_abs_sec": 0.2,
                    "low_confidence_ratio": 0.01,
                    "dtw_line_coverage": 0.95,
                },
            },
            {
                "artist": "B",
                "title": "Bad",
                "status": "ok",
                "metrics": {
                    "agreement_coverage_ratio": 0.15,
                    "agreement_start_p95_abs_sec": 1.3,
                    "low_confidence_ratio": 0.15,
                    "dtw_line_coverage": 0.7,
                    "fallback_map_attempted": 2,
                    "fallback_map_selected": 0,
                },
            },
        ]
    }
    (run_dir / "benchmark_report.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recommend_human_guidance_tasks.py",
            "--report",
            str(run_dir),
            "--top",
            "1",
        ],
    )
    rc = _MODULE.main()
    assert rc == 0
    payload = json.loads((run_dir / "human_guidance_tasks.json").read_text("utf-8"))
    assert payload["song_count_considered"] == 1
    assert payload["rows"][0]["song"] == "B - Bad"
    assert payload["rows"][0]["priority_score"] > 0.0
    md = (run_dir / "human_guidance_tasks.md").read_text("utf-8")
    assert "Human Guidance Task Recommendations" in md
