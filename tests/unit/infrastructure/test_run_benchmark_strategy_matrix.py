import importlib.util
import json
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "run_benchmark_strategy_matrix.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_benchmark_strategy_matrix_module", _MODULE_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_build_command_includes_strategy_and_flags(tmp_path):
    cmd = _MODULE._build_command(
        python_bin="python",
        strategy="whisper_only",
        run_id="run-1",
        output_root=tmp_path / "results",
        manifest=tmp_path / "manifest.yaml",
        gold_root=tmp_path / "gold",
        cache_dir=tmp_path / "cache",
        offline=True,
        force=False,
        max_songs=2,
        match="Hero",
        scenario="lyrics_no_timing",
    )
    assert "--strategy" in cmd
    assert "whisper_only" in cmd
    assert "--offline" in cmd
    assert "--max-songs" in cmd
    assert "2" in cmd
    assert "--match" in cmd
    assert "--scenario" in cmd
    assert "lyrics_no_timing" in cmd


def test_extract_summary_handles_aggregate_fields():
    report = {
        "status": "finished",
        "aggregate": {
            "songs_total": 3,
            "songs_succeeded": 2,
            "songs_failed": 1,
            "dtw_line_coverage_line_weighted_mean": 0.7,
            "dtw_word_coverage_line_weighted_mean": 0.6,
            "agreement_start_mean_abs_sec_mean": 0.42,
            "agreement_start_p95_abs_sec_mean": 0.9,
            "low_confidence_ratio_total": 0.1,
            "avg_abs_word_start_delta_sec_word_weighted_mean": 1.23,
        },
    }
    summary = _MODULE._extract_summary(report)
    assert summary["status"] == "finished"
    assert summary["songs_total"] == 3
    assert summary["dtw_line_coverage_line_weighted_mean"] == 0.7
    assert summary["agreement_start_mean_abs_sec_line_weighted_mean"] == 0.42
    assert summary["low_confidence_ratio_line_weighted_mean"] == 0.1
    assert summary["gold_start_abs_word_weighted_mean"] == 1.23
    assert summary["sum_song_elapsed_sec"] is None


def test_extract_summary_preserves_missing_independent_agreement():
    report = {
        "status": "finished_with_warnings",
        "aggregate": {
            "songs_total": 1,
            "songs_succeeded": 1,
            "agreement_start_mean_abs_sec_mean": None,
            "agreement_start_p95_abs_sec_mean": None,
            "whisper_anchor_start_mean_abs_sec_mean": 0.4,
            "whisper_anchor_start_p95_abs_sec_mean": 0.9,
        },
    }
    summary = _MODULE._extract_summary(report)
    assert summary["agreement_start_mean_abs_sec_line_weighted_mean"] is None
    assert summary["agreement_start_p95_abs_sec_line_weighted_mean"] is None
    assert summary["whisper_anchor_start_mean_abs_sec_mean"] == 0.4


def test_write_markdown(tmp_path):
    out = tmp_path / "report.md"
    _MODULE._write_markdown(
        out,
        [
            {
                "strategy": "hybrid_dtw",
                "status": "finished",
                "songs_succeeded": 1,
                "songs_total": 2,
                "dtw_line_coverage_line_weighted_mean": 0.8,
                "dtw_word_coverage_line_weighted_mean": 0.7,
                "agreement_start_mean_abs_sec_line_weighted_mean": 0.3,
                "agreement_start_p95_abs_sec_line_weighted_mean": 0.6,
                "low_confidence_ratio_line_weighted_mean": 0.05,
                "sum_song_elapsed_sec": 120.0,
            }
        ],
    )
    text = out.read_text(encoding="utf-8")
    assert "Benchmark Strategy Matrix" in text
    assert "hybrid_dtw" in text


def test_main_writes_reports_for_successful_mock_runs(tmp_path, monkeypatch):
    output_root = tmp_path / "results"

    class DummyProc:
        returncode = 0

    def fake_run(cmd):
        run_id = cmd[cmd.index("--run-id") + 1]
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "benchmark_report.json").write_text(
            json.dumps(
                {
                    "status": "finished",
                    "aggregate": {"songs_total": 1, "songs_succeeded": 1},
                }
            ),
            encoding="utf-8",
        )
        return DummyProc()

    monkeypatch.setattr(_MODULE.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_benchmark_strategy_matrix.py",
            "--output-root",
            str(output_root),
            "--matrix-id",
            "mx-1",
            "--strategies",
            "hybrid_dtw,whisper_only",
        ],
    )

    assert _MODULE.main() == 0
    matrix_json = output_root / "mx-1-matrix" / "strategy_matrix_report.json"
    assert matrix_json.exists()
    payload = json.loads(matrix_json.read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 2
    assert "recommendations" in payload


def test_recommendations_select_expected_strategy():
    rows = [
        {
            "strategy": "hybrid_dtw",
            "agreement_start_p95_abs_sec_line_weighted_mean": 0.5,
            "agreement_start_mean_abs_sec_line_weighted_mean": 0.2,
            "low_confidence_ratio_line_weighted_mean": 0.1,
            "dtw_line_coverage_line_weighted_mean": 0.8,
            "sum_song_elapsed_sec": 200.0,
        },
        {
            "strategy": "whisper_only",
            "agreement_start_p95_abs_sec_line_weighted_mean": 0.7,
            "agreement_start_mean_abs_sec_line_weighted_mean": 0.35,
            "low_confidence_ratio_line_weighted_mean": 0.2,
            "dtw_line_coverage_line_weighted_mean": 0.6,
            "sum_song_elapsed_sec": 100.0,
        },
    ]
    rec = _MODULE._recommendations(rows)
    assert rec["best_p95_start_abs_sec"] == "hybrid_dtw"
    assert rec["best_mean_start_abs_sec"] == "hybrid_dtw"
    assert rec["lowest_low_confidence_ratio"] == "hybrid_dtw"
    assert rec["highest_dtw_line_coverage"] == "hybrid_dtw"
    assert rec["fastest_runtime"] == "whisper_only"
    assert rec["best_quality_runtime_balance"] == "hybrid_dtw"
