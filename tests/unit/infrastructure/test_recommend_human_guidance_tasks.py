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
    assert payload["min_priority"] == 0.0
    assert payload["rows"][0]["song"] == "B - Bad"
    assert payload["rows"][0]["priority_score"] > 0.0
    md = (run_dir / "human_guidance_tasks.md").read_text("utf-8")
    assert "Human Guidance Task Recommendations" in md


def test_recommend_human_guidance_tasks_filters_by_min_priority(
    tmp_path, monkeypatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report = {
        "songs": [
            {
                "artist": "A",
                "title": "Low",
                "status": "ok",
                "metrics": {
                    "agreement_coverage_ratio": 0.9,
                    "agreement_start_p95_abs_sec": 0.1,
                    "low_confidence_ratio": 0.0,
                    "dtw_line_coverage": 1.0,
                },
            },
            {
                "artist": "B",
                "title": "High",
                "status": "ok",
                "metrics": {
                    "agreement_coverage_ratio": 0.1,
                    "agreement_start_p95_abs_sec": 1.4,
                    "low_confidence_ratio": 0.2,
                    "dtw_line_coverage": 0.6,
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
            "--min-priority",
            "2.0",
            "--top",
            "10",
        ],
    )
    rc = _MODULE.main()
    assert rc == 0
    payload = json.loads((run_dir / "human_guidance_tasks.json").read_text("utf-8"))
    assert payload["song_count_considered"] == 1
    assert payload["rows"][0]["song"] == "B - High"


def test_recommend_human_guidance_tasks_includes_mismatch_examples(
    tmp_path, monkeypatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    timing_report = run_dir / "song_timing_report.json"
    timing_report.write_text(
        json.dumps(
            {
                "lines": [
                    {
                        "index": 0,
                        "start": 10.0,
                        "nearest_segment_start": 10.1,
                        "text": "si el ritmo te lleva a mover la cabeza",
                        "nearest_segment_start_text": "cabeza mover lleva ritmo te el si la",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report = {
        "songs": [
            {
                "artist": "C",
                "title": "Mismatch",
                "status": "ok",
                "report_path": str(timing_report),
                "metrics": {
                    "agreement_coverage_ratio": 0.2,
                    "agreement_start_p95_abs_sec": 1.0,
                    "low_confidence_ratio": 0.02,
                    "dtw_line_coverage": 1.0,
                },
            }
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
            "10",
        ],
    )
    rc = _MODULE.main()
    assert rc == 0
    payload = json.loads((run_dir / "human_guidance_tasks.json").read_text("utf-8"))
    examples = payload["rows"][0]["mismatch_examples"]
    assert len(examples) == 1
    assert "delta=0.10s" in examples[0]
    targets = payload["rows"][0]["suggested_targets"]
    assert len(targets) == 1
    assert "line_index=0" in targets[0]
    assert "likely_lexical_mismatch" in targets[0]
    md = (run_dir / "human_guidance_tasks.md").read_text("utf-8")
    assert "example mismatch" in md
    assert "suggested target" in md
