import importlib.util
import json
from pathlib import Path
import sys

_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "profile_benchmark_runtime.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "profile_benchmark_runtime_module", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_runtime_profile_orders_by_elapsed(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report = {
        "songs": [
            {
                "artist": "A",
                "title": "Fast",
                "elapsed_sec": 2.0,
                "status": "ok",
                "phase_durations_sec": {"alignment": 1.0},
                "metrics": {"fallback_map_attempted": 0, "fallback_map_selected": 0},
            },
            {
                "artist": "B",
                "title": "Slow",
                "elapsed_sec": 9.5,
                "status": "ok",
                "phase_durations_sec": {"alignment": 6.0, "whisper": 2.0},
                "metrics": {
                    "fallback_map_attempted": 1,
                    "fallback_map_selected": 1,
                    "local_transcribe_cache_hits": 1,
                    "local_transcribe_cache_misses": 1,
                },
            },
            {"artist": "C", "title": "Mid", "elapsed_sec": 4.0, "status": "ok"},
        ]
    }
    (run_dir / "benchmark_report.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_benchmark_runtime.py",
            "--report",
            str(run_dir),
            "--top",
            "2",
        ],
    )
    rc = _MODULE.main()
    assert rc == 0
    out_json = json.loads(
        (run_dir / "runtime_profile.json").read_text(encoding="utf-8")
    )
    assert out_json["rows"][0]["song"] == "B - Slow"
    assert out_json["rows"][1]["song"] == "C - Mid"
    assert out_json["rows"][0]["bottleneck_phase"] == "alignment"
    assert out_json["rows"][0]["fallback_map_attempted"] == 1
    assert out_json["rows"][0]["local_transcribe_cache_hits"] == 1
    assert (run_dir / "runtime_profile.md").exists()
