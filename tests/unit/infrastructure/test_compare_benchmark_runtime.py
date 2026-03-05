import importlib.util
import json
from pathlib import Path
import sys

_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "compare_benchmark_runtime.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "compare_benchmark_runtime_module", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_compare_runtime_reports_writes_outputs(tmp_path, monkeypatch) -> None:
    base = tmp_path / "base"
    cand = tmp_path / "cand"
    base.mkdir()
    cand.mkdir()
    baseline_doc = {
        "suite_wall_elapsed_sec": 10.0,
        "aggregate": {"sum_song_elapsed_sec": 9.0},
        "songs": [
            {
                "artist": "A",
                "title": "Song",
                "elapsed_sec": 5.0,
                "phase_durations_sec": {"alignment": 4.0},
                "metrics": {"fallback_map_attempted": 0},
            }
        ],
    }
    candidate_doc = {
        "suite_wall_elapsed_sec": 12.0,
        "aggregate": {"sum_song_elapsed_sec": 11.0},
        "songs": [
            {
                "artist": "A",
                "title": "Song",
                "elapsed_sec": 6.5,
                "phase_durations_sec": {"alignment": 5.0},
                "metrics": {"fallback_map_attempted": 1},
            }
        ],
    }
    (base / "benchmark_report.json").write_text(
        json.dumps(baseline_doc), encoding="utf-8"
    )
    (cand / "benchmark_report.json").write_text(
        json.dumps(candidate_doc), encoding="utf-8"
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_benchmark_runtime.py",
            "--baseline",
            str(base),
            "--candidate",
            str(cand),
        ],
    )
    rc = _MODULE.main()
    assert rc == 0
    out_json = json.loads((cand / "runtime_delta.json").read_text(encoding="utf-8"))
    assert out_json["common_song_count"] == 1
    assert out_json["rows_emitted"] == 1
    assert out_json["filters"] == {"top": 0, "only_positive_delta": False}
    assert out_json["alignment_delta_comparable_song_count"] == 1
    assert out_json["whisper_delta_comparable_song_count"] == 0
    assert out_json["suite_wall_elapsed_delta_sec"] == 2.0
    assert out_json["sum_song_elapsed_executed_delta_sec"] == 2.0
    assert out_json["sum_song_elapsed_total_delta_sec"] == 2.0
    assert out_json["warnings"] == [
        "Whisper phase deltas are partially non-comparable across reports."
    ]
    assert out_json["rows"][0]["elapsed_delta_sec"] == 1.5
    assert (cand / "runtime_delta.md").exists()


def test_compare_runtime_reports_handles_missing_phase_durations(
    tmp_path, monkeypatch
) -> None:
    base = tmp_path / "base"
    cand = tmp_path / "cand"
    base.mkdir()
    cand.mkdir()
    baseline_doc = {
        "suite_wall_elapsed_sec": 10.0,
        "aggregate": {"sum_song_elapsed_sec": 9.0},
        "songs": [{"artist": "A", "title": "Song", "elapsed_sec": 5.0}],
    }
    candidate_doc = {
        "suite_wall_elapsed_sec": 11.0,
        "aggregate": {"sum_song_elapsed_sec": 10.0},
        "songs": [
            {
                "artist": "A",
                "title": "Song",
                "elapsed_sec": 5.8,
                "phase_durations_sec": {"alignment": 4.0, "whisper": 1.2},
            }
        ],
    }
    (base / "benchmark_report.json").write_text(
        json.dumps(baseline_doc), encoding="utf-8"
    )
    (cand / "benchmark_report.json").write_text(
        json.dumps(candidate_doc), encoding="utf-8"
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_benchmark_runtime.py",
            "--baseline",
            str(base),
            "--candidate",
            str(cand),
        ],
    )
    rc = _MODULE.main()
    assert rc == 0
    out_json = json.loads((cand / "runtime_delta.json").read_text(encoding="utf-8"))
    row = out_json["rows"][0]
    assert row["alignment_delta_sec"] is None
    assert row["whisper_delta_sec"] is None
    assert out_json["alignment_delta_comparable_song_count"] == 0
    assert out_json["whisper_delta_comparable_song_count"] == 0
    out_md = (cand / "runtime_delta.md").read_text(encoding="utf-8")
    assert "n/a" in out_md


def test_compare_runtime_reports_uses_total_elapsed_when_available(
    tmp_path, monkeypatch
) -> None:
    base = tmp_path / "base"
    cand = tmp_path / "cand"
    base.mkdir()
    cand.mkdir()
    baseline_doc = {
        "suite_wall_elapsed_sec": 100.0,
        "aggregate": {"sum_song_elapsed_sec": 90.0, "sum_song_elapsed_total_sec": 95.0},
        "songs": [{"artist": "A", "title": "Song", "elapsed_sec": 10.0}],
    }
    candidate_doc = {
        "suite_wall_elapsed_sec": 102.0,
        "aggregate": {"sum_song_elapsed_sec": 1.0, "sum_song_elapsed_total_sec": 97.0},
        "songs": [{"artist": "A", "title": "Song", "elapsed_sec": 9.0}],
    }
    (base / "benchmark_report.json").write_text(
        json.dumps(baseline_doc), encoding="utf-8"
    )
    (cand / "benchmark_report.json").write_text(
        json.dumps(candidate_doc), encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_benchmark_runtime.py",
            "--baseline",
            str(base),
            "--candidate",
            str(cand),
        ],
    )
    rc = _MODULE.main()
    assert rc == 0
    out_json = json.loads((cand / "runtime_delta.json").read_text(encoding="utf-8"))
    assert out_json["sum_song_elapsed_executed_delta_sec"] == -89.0
    assert out_json["sum_song_elapsed_total_delta_sec"] == 2.0
    assert (
        "Executed vs total elapsed deltas diverge; one report may be aggregate-only."
        in out_json["warnings"]
    )


def test_compare_runtime_reports_supports_top_and_positive_filter(
    tmp_path, monkeypatch
) -> None:
    base = tmp_path / "base"
    cand = tmp_path / "cand"
    base.mkdir()
    cand.mkdir()
    baseline_doc = {
        "suite_wall_elapsed_sec": 10.0,
        "aggregate": {"sum_song_elapsed_sec": 9.0},
        "songs": [
            {"artist": "A", "title": "Fast", "elapsed_sec": 4.0},
            {"artist": "A", "title": "Slow", "elapsed_sec": 5.0},
        ],
    }
    candidate_doc = {
        "suite_wall_elapsed_sec": 10.5,
        "aggregate": {"sum_song_elapsed_sec": 9.5},
        "songs": [
            {"artist": "A", "title": "Fast", "elapsed_sec": 3.0},
            {"artist": "A", "title": "Slow", "elapsed_sec": 6.0},
        ],
    }
    (base / "benchmark_report.json").write_text(
        json.dumps(baseline_doc), encoding="utf-8"
    )
    (cand / "benchmark_report.json").write_text(
        json.dumps(candidate_doc), encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_benchmark_runtime.py",
            "--baseline",
            str(base),
            "--candidate",
            str(cand),
            "--only-positive-delta",
            "--top",
            "1",
        ],
    )
    rc = _MODULE.main()
    assert rc == 0
    out_json = json.loads((cand / "runtime_delta.json").read_text(encoding="utf-8"))
    assert out_json["rows_emitted"] == 1
    assert out_json["filters"] == {"top": 1, "only_positive_delta": True}
    assert len(out_json["rows"]) == 1
    assert out_json["rows"][0]["song"] == "A - Slow"
