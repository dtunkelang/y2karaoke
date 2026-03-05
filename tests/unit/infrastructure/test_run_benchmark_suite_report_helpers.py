"""Unit tests for benchmark suite report helper functions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[3] / "tools" / "run_benchmark_suite.py"
    )
    spec = importlib.util.spec_from_file_location("run_benchmark_suite", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        offline=False,
        force=True,
        strategy="hybrid_whisper",
        scenario="default",
        no_whisper_map_lrc_dtw=False,
        timeout_sec=1800,
        heartbeat_sec=30,
        match="",
        max_songs=0,
        min_dtw_song_coverage_ratio=0.9,
        min_dtw_line_coverage_ratio=0.9,
        min_timing_quality_score_line_weighted=0.58,
        strict_quality_coverage=False,
        agreement_baseline_report=None,
        min_agreement_coverage_gain_for_bad_ratio_warning=0.0,
        max_agreement_bad_ratio_increase_on_coverage_gain=0.0,
        strict_agreement_tradeoff=False,
        expect_cached_separation=False,
        expect_cached_whisper=False,
        strict_cache_expectations=False,
        max_whisper_phase_share=0.0,
        max_alignment_phase_share=0.0,
        max_scheduler_overhead_sec=0.0,
        strict_runtime_budgets=False,
        rebaseline=False,
        gold_root=tmp_path,
        aggregate_only=False,
    )


def test_compute_suite_elapsed_accounting_aggregate_only_uses_total():
    module = _load_module()
    aggregate = {"sum_song_elapsed_sec": 10.0, "sum_song_elapsed_total_sec": 12.5}
    suite_elapsed, sum_song, sum_song_total, overhead_base = (
        module._compute_suite_elapsed_accounting(
            aggregate=aggregate,
            measured_suite_elapsed=8.0,
            aggregate_only=True,
        )
    )
    assert suite_elapsed == 12.5
    assert sum_song == 10.0
    assert sum_song_total == 12.5
    assert overhead_base == 12.5


def test_build_final_report_options_includes_runtime_and_skip_count(tmp_path):
    module = _load_module()
    options = module._build_final_report_options(
        _sample_args(tmp_path), aggregate_only_skipped_missing_count=3
    )
    assert options["force"] is True
    assert options["whisper_map_lrc_dtw"] is True
    assert options["max_whisper_phase_share"] == 0.0
    assert options["max_alignment_phase_share"] == 0.0
    assert options["max_scheduler_overhead_sec"] == 0.0
    assert options["strict_runtime_budgets"] is False
    assert options["aggregate_only_skipped_missing_count"] == 3


def test_build_final_report_json_marks_finished_with_warnings(tmp_path):
    module = _load_module()
    payload = module._build_final_report_json(
        run_id="run123",
        manifest_path=tmp_path / "manifest.yaml",
        args=_sample_args(tmp_path),
        aggregate={"songs_total": 1, "songs_failed": 0},
        song_results=[{"status": "ok"}],
        suite_elapsed=22.0,
        sum_song_elapsed=20.0,
        sum_song_elapsed_total=20.0,
        overhead_base=20.0,
        quality_warnings=["qwarn"],
        cache_warnings=[],
        runtime_warnings=[],
        run_warnings=["qwarn"],
        aggregate_only_skipped_missing=["A - Song"],
    )
    assert payload["status"] == "finished_with_warnings"
    assert payload["scheduler_overhead_sec"] == 2.0
    assert payload["options"]["aggregate_only_skipped_missing_count"] == 1
    assert payload["aggregate_only_skipped_missing_songs"] == ["A - Song"]


def test_load_baseline_aggregate_supports_run_dir_and_missing_aggregate(tmp_path):
    module = _load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "benchmark_report.json").write_text(
        json.dumps({"aggregate": {"agreement_coverage_ratio_mean": 0.2}}),
        encoding="utf-8",
    )
    agg = module._load_baseline_aggregate(run_dir)
    assert agg == {"agreement_coverage_ratio_mean": 0.2}

    report_file = tmp_path / "report.json"
    report_file.write_text(json.dumps({"status": "finished"}), encoding="utf-8")
    assert module._load_baseline_aggregate(report_file) is None
    assert module._load_baseline_aggregate(None) is None


def test_build_runner_env_prefixes_repo_src(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("PYTHONPATH", "alpha:beta")
    env = module._build_runner_env()
    assert env["PYTHONPATH"].startswith(f"{module.REPO_ROOT / 'src'}:")
    assert env["PYTHONPATH"].endswith("alpha:beta")


def test_collect_song_results_aggregate_only_delegates(tmp_path):
    module = _load_module()
    songs = [
        module.BenchmarkSong(
            artist="Artist A",
            title="Alpha",
            youtube_id="aaaaaaaaaaa",
            youtube_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
        )
    ]
    args = _sample_args(tmp_path)
    args.aggregate_only = True
    args.rebaseline = True

    calls: list[dict[str, object]] = []

    def fake_load_aggregate_only_results(**kwargs):
        calls.append(kwargs)
        return ([{"status": "ok"}], ["missing"])

    old_fn = module._load_aggregate_only_results
    module._load_aggregate_only_results = fake_load_aggregate_only_results
    try:
        song_results, skipped = module._collect_song_results(
            args=args,
            songs=songs,
            run_id="run-1",
            run_dir=tmp_path,
            manifest_path=tmp_path / "manifest.yaml",
            run_signature={"k": "v"},
            gold_root=tmp_path,
            env=os.environ.copy(),
            suite_start=0.0,
        )
    finally:
        module._load_aggregate_only_results = old_fn

    assert song_results == [{"status": "ok"}]
    assert skipped == ["missing"]
    assert len(calls) == 1
    assert calls[0]["songs"] == songs
    assert calls[0]["run_dir"] == tmp_path
    assert calls[0]["gold_root"] == tmp_path
    assert calls[0]["rebaseline"] is True


def test_derive_run_status_and_exit_code_policy(tmp_path):
    module = _load_module()
    aggregate_ok = {"songs_failed": 0}
    aggregate_fail = {"songs_failed": 1}
    assert module._derive_run_status(aggregate_ok, []) == "OK"
    assert module._derive_run_status(aggregate_ok, ["warn"]) == "WARN"
    assert module._derive_run_status(aggregate_fail, []) == "FAIL"

    args = _sample_args(tmp_path)
    args.strict_quality_coverage = True
    args.strict_agreement_tradeoff = True
    args.strict_cache_expectations = True
    args.strict_runtime_budgets = True
    assert (
        module._determine_exit_code(
            aggregate=aggregate_fail,
            args=args,
            quality_warnings=[],
            agreement_tradeoff_warnings=[],
            cache_warnings=[],
            runtime_warnings=[],
        )
        == 2
    )
    assert (
        module._determine_exit_code(
            aggregate=aggregate_ok,
            args=args,
            quality_warnings=["q"],
            agreement_tradeoff_warnings=[],
            cache_warnings=[],
            runtime_warnings=[],
        )
        == 3
    )
    assert (
        module._determine_exit_code(
            aggregate=aggregate_ok,
            args=args,
            quality_warnings=[],
            agreement_tradeoff_warnings=["a"],
            cache_warnings=[],
            runtime_warnings=[],
        )
        == 6
    )
    assert (
        module._determine_exit_code(
            aggregate=aggregate_ok,
            args=args,
            quality_warnings=[],
            agreement_tradeoff_warnings=[],
            cache_warnings=["c"],
            runtime_warnings=[],
        )
        == 4
    )
    assert (
        module._determine_exit_code(
            aggregate=aggregate_ok,
            args=args,
            quality_warnings=[],
            agreement_tradeoff_warnings=[],
            cache_warnings=[],
            runtime_warnings=["r"],
        )
        == 5
    )
    args.strict_runtime_budgets = False
    assert (
        module._determine_exit_code(
            aggregate=aggregate_ok,
            args=args,
            quality_warnings=[],
            agreement_tradeoff_warnings=[],
            cache_warnings=[],
            runtime_warnings=["r"],
        )
        == 0
    )


def test_compute_run_warnings_includes_aggregate_only_skip_message(tmp_path):
    module = _load_module()
    args = _sample_args(tmp_path)
    old_quality = module._quality_coverage_warnings
    old_agreement = module._agreement_tradeoff_warnings
    old_cache = module._cache_expectation_warnings
    old_runtime = module._runtime_budget_warnings
    module._quality_coverage_warnings = lambda **_: ["q1"]  # type: ignore[assignment]
    module._agreement_tradeoff_warnings = lambda **_: ["a1"]  # type: ignore[assignment]
    module._cache_expectation_warnings = lambda **_: ["c1"]  # type: ignore[assignment]
    module._runtime_budget_warnings = lambda **_: ["r1"]  # type: ignore[assignment]
    try:
        quality, agreement, cache, runtime, run = module._compute_run_warnings(
            args=args,
            aggregate={"songs_failed": 0},
            suite_elapsed=1.0,
            baseline_aggregate=None,
            aggregate_only_skipped_missing=["A - Song", "B - Song"],
        )
    finally:
        module._quality_coverage_warnings = old_quality
        module._agreement_tradeoff_warnings = old_agreement
        module._cache_expectation_warnings = old_cache
        module._runtime_budget_warnings = old_runtime

    assert quality == ["q1", "a1"]
    assert agreement == ["a1"]
    assert cache == ["c1"]
    assert runtime == ["r1"]
    assert run[-1] == "Aggregate-only skipped songs without cached results: 2"


def test_persist_final_report_outputs_writes_json_md_progress_and_latest(tmp_path):
    module = _load_module()
    run_dir = tmp_path / "run"
    out_root = tmp_path / "out"
    run_dir.mkdir()
    out_root.mkdir()
    args = _sample_args(tmp_path)
    args.output_root = out_root

    wrote_markdown: list[tuple[Path, str]] = []

    def fake_write_markdown_summary(path, **kwargs):
        wrote_markdown.append((path, kwargs["run_id"]))
        path.write_text("# md\n", encoding="utf-8")

    old_write_markdown_summary = module._write_markdown_summary
    module._write_markdown_summary = fake_write_markdown_summary
    try:
        json_path, md_path = module._persist_final_report_outputs(
            args=args,
            run_id="run-xyz",
            run_dir=run_dir,
            manifest_path=tmp_path / "manifest.yaml",
            aggregate={"songs_total": 1},
            song_results=[{"status": "ok"}],
            report_json={"run_id": "run-xyz", "status": "finished"},
        )
    finally:
        module._write_markdown_summary = old_write_markdown_summary

    assert json_path.exists()
    assert md_path.exists()
    assert (run_dir / "benchmark_progress.json").exists()
    latest = (out_root / "latest.json").read_text(encoding="utf-8").strip()
    assert latest == str(json_path)
    assert wrote_markdown == [(md_path, "run-xyz")]


def test_prepare_run_context_resolves_manifest_run_dir_and_baseline(tmp_path):
    module = _load_module()
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        (
            "songs:\n"
            "  - artist: Artist A\n"
            "    title: Alpha\n"
            "    youtube_id: aaaaaaaaaaa\n"
            "    youtube_url: https://www.youtube.com/watch?v=aaaaaaaaaaa\n"
            "  - artist: Artist B\n"
            "    title: Beta\n"
            "    youtube_id: bbbbbbbbbbb\n"
            "    youtube_url: https://www.youtube.com/watch?v=bbbbbbbbbbb\n"
        ),
        encoding="utf-8",
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps({"aggregate": {"agreement_coverage_ratio_mean": 0.2}}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        manifest=manifest_path,
        match="Beta",
        max_songs=1,
        output_root=tmp_path / "results",
        run_id="run-fixed",
        resume_run_dir=None,
        resume_latest=False,
        aggregate_only=False,
        gold_root=tmp_path / "gold",
        agreement_baseline_report=baseline_path,
        offline=False,
        force=False,
        strategy="hybrid_whisper",
        scenario="default",
        no_whisper_map_lrc_dtw=False,
        cache_dir=None,
        min_timing_quality_score_line_weighted=0.58,
    )
    (
        resolved_manifest,
        songs,
        run_dir,
        run_id,
        env,
        run_signature,
        gold_root,
        baseline_aggregate,
    ) = module._prepare_run_context(args)

    assert resolved_manifest == manifest_path.resolve()
    assert [song.title for song in songs] == ["Beta"]
    assert run_dir == (tmp_path / "results" / "run-fixed")
    assert run_dir.exists()
    assert run_id == "run-fixed"
    assert env["PYTHONPATH"].startswith(str(module.REPO_ROOT / "src"))
    assert run_signature["strategy"] == "hybrid_whisper"
    assert gold_root == (tmp_path / "gold").resolve()
    assert baseline_aggregate == {"agreement_coverage_ratio_mean": 0.2}


def test_try_reuse_cached_song_result_reuses_matching_ok_result(tmp_path):
    module = _load_module()
    song = module.BenchmarkSong(
        artist="Artist A",
        title="Alpha",
        youtube_id="aaaaaaaaaaa",
        youtube_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
    )
    args = argparse.Namespace(
        reuse_mismatched_results=False,
        rerun_completed=False,
        rerun_failed=False,
        rebaseline=False,
    )
    old_load = module._load_song_result
    old_refresh = module._refresh_cached_metrics
    module._load_song_result = lambda _: {"status": "ok", "run_signature": {"k": "v"}}  # type: ignore[assignment]
    module._refresh_cached_metrics = (  # type: ignore[assignment]
        lambda prior, **_: {**prior, "metrics": {"dtw_line_coverage": 1.0}}
    )
    try:
        reused = module._try_reuse_cached_song_result(
            args=args,
            run_signature={"k": "v"},
            index=1,
            total_songs=1,
            song=song,
            result_path=tmp_path / "result.json",
            report_path=tmp_path / "report.json",
            gold_root=tmp_path,
        )
    finally:
        module._load_song_result = old_load
        module._refresh_cached_metrics = old_refresh
    assert reused is not None
    assert reused["result_reused"] is True
    assert reused["metrics"]["dtw_line_coverage"] == 1.0


def test_try_reuse_cached_song_result_ignores_mismatched_signature(tmp_path):
    module = _load_module()
    song = module.BenchmarkSong(
        artist="Artist A",
        title="Alpha",
        youtube_id="aaaaaaaaaaa",
        youtube_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
    )
    args = argparse.Namespace(
        reuse_mismatched_results=False,
        rerun_completed=False,
        rerun_failed=False,
        rebaseline=False,
    )
    old_load = module._load_song_result
    module._load_song_result = lambda _: {"status": "ok", "run_signature": {"old": 1}}  # type: ignore[assignment]
    try:
        reused = module._try_reuse_cached_song_result(
            args=args,
            run_signature={"new": 1},
            index=1,
            total_songs=1,
            song=song,
            result_path=tmp_path / "result.json",
            report_path=tmp_path / "report.json",
            gold_root=tmp_path,
        )
    finally:
        module._load_song_result = old_load
    assert reused is None


def test_run_single_song_generation_returns_record_and_result_path(tmp_path):
    module = _load_module()
    song = module.BenchmarkSong(
        artist="Artist A",
        title="Alpha",
        youtube_id="aaaaaaaaaaa",
        youtube_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
    )
    args = argparse.Namespace(
        python_bin="python",
        cache_dir=None,
        offline=False,
        force=False,
        no_whisper_map_lrc_dtw=False,
        strategy="hybrid_whisper",
        scenario="default",
        timeout_sec=30,
        heartbeat_sec=1,
        rebaseline=False,
    )
    old_build_cmd = module._build_generate_command
    old_load_gold = module._load_gold_doc
    old_run_cmd = module._run_song_command
    module._build_generate_command = lambda **_: ["python", "-m", "noop"]  # type: ignore[assignment]
    module._load_gold_doc = lambda **_: None  # type: ignore[assignment]
    module._run_song_command = (  # type: ignore[assignment]
        lambda **_: {"status": "ok", "elapsed_sec": 1.23}
    )
    try:
        record, result_path = module._run_single_song_generation(
            args=args,
            index=1,
            total_songs=1,
            song=song,
            run_dir=tmp_path,
            run_signature={"k": "v"},
            gold_root=tmp_path,
            env={},
        )
    finally:
        module._build_generate_command = old_build_cmd
        module._load_gold_doc = old_load_gold
        module._run_song_command = old_run_cmd
    assert record["status"] == "ok"
    assert result_path.name.endswith(f"{song.slug}_result.json")


def test_append_result_and_checkpoint_writes_optional_result_file(tmp_path):
    module = _load_module()
    args = _sample_args(tmp_path)
    song_results: list[dict[str, object]] = []
    writes: list[Path] = []
    checkpoints: list[int] = []

    old_write_json = module._write_json
    old_write_checkpoint = module._write_checkpoint
    module._write_json = lambda path, payload: writes.append(path)  # type: ignore[assignment]
    module._write_checkpoint = (  # type: ignore[assignment]
        lambda **kwargs: checkpoints.append(len(kwargs["song_results"]))
    )
    try:
        module._append_result_and_checkpoint(
            record={"status": "ok"},
            song_results=song_results,
            run_id="run",
            run_dir=tmp_path,
            manifest_path=tmp_path / "manifest.yaml",
            args=args,
            suite_start=0.0,
            result_path=tmp_path / "song_result.json",
        )
        module._append_result_and_checkpoint(
            record={"status": "failed"},
            song_results=song_results,
            run_id="run",
            run_dir=tmp_path,
            manifest_path=tmp_path / "manifest.yaml",
            args=args,
            suite_start=0.0,
            result_path=None,
        )
    finally:
        module._write_json = old_write_json
        module._write_checkpoint = old_write_checkpoint

    assert len(song_results) == 2
    assert writes == [tmp_path / "song_result.json"]
    assert checkpoints == [1, 2]


def test_collect_single_song_result_prefers_reuse_path(tmp_path):
    module = _load_module()
    song = module.BenchmarkSong(
        artist="Artist A",
        title="Alpha",
        youtube_id="aaaaaaaaaaa",
        youtube_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
    )
    args = argparse.Namespace()
    song_results: list[dict[str, object]] = []
    old_try_reuse = module._try_reuse_cached_song_result
    old_run_single = module._run_single_song_generation
    old_append = module._append_result_and_checkpoint
    module._try_reuse_cached_song_result = lambda **_: {  # type: ignore[assignment]
        "status": "ok",
        "result_reused": True,
    }
    module._run_single_song_generation = lambda **_: (_ for _ in ()).throw(  # type: ignore[assignment]
        AssertionError("should not run generation when reused")
    )
    module._append_result_and_checkpoint = (  # type: ignore[assignment]
        lambda **kwargs: kwargs["song_results"].append(kwargs["record"])
    )
    try:
        record = module._collect_single_song_result(
            args=args,
            song_results=song_results,
            index=1,
            total_songs=1,
            song=song,
            run_signature={},
            run_id="run",
            run_dir=tmp_path,
            manifest_path=tmp_path / "manifest.yaml",
            gold_root=tmp_path,
            env={},
            suite_start=0.0,
        )
    finally:
        module._try_reuse_cached_song_result = old_try_reuse
        module._run_single_song_generation = old_run_single
        module._append_result_and_checkpoint = old_append

    assert record["result_reused"] is True
    assert len(song_results) == 1


def test_collect_song_results_honors_fail_fast(tmp_path):
    module = _load_module()
    songs = [
        module.BenchmarkSong(
            artist="Artist A",
            title="Alpha",
            youtube_id="aaaaaaaaaaa",
            youtube_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
        ),
        module.BenchmarkSong(
            artist="Artist B",
            title="Beta",
            youtube_id="bbbbbbbbbbb",
            youtube_url="https://www.youtube.com/watch?v=bbbbbbbbbbb",
        ),
    ]
    args = argparse.Namespace(
        aggregate_only=False,
        rebaseline=False,
        fail_fast=True,
    )
    old_collect_single = module._collect_single_song_result
    calls: list[int] = []
    module._collect_single_song_result = lambda **kwargs: (  # type: ignore[assignment]
        calls.append(kwargs["index"]) or {"status": "failed", "result_reused": False}
    )
    try:
        rows, skipped = module._collect_song_results(
            args=args,
            songs=songs,
            run_id="run",
            run_dir=tmp_path,
            manifest_path=tmp_path / "manifest.yaml",
            run_signature={},
            gold_root=tmp_path,
            env={},
            suite_start=0.0,
        )
    finally:
        module._collect_single_song_result = old_collect_single

    assert calls == [1]
    assert len(rows) == 0  # helper was mocked and didn't append into rows
    assert skipped == []


def test_expand_agreement_token_handles_known_patterns():
    module = _load_module()
    assert module._expand_agreement_token("gonna") == ["going", "to"]
    assert module._expand_agreement_token("can't") == ["can", "not"]
    assert module._expand_agreement_token("walkin'") == ["walking"]
    assert module._expand_agreement_token("we're") == ["we", "are"]
    assert module._expand_agreement_token("they've") == ["they", "have"]
    assert module._expand_agreement_token("i'll") == ["i", "will"]


def test_infer_reference_divergence_no_gold_branch():
    module = _load_module()
    result = module._infer_reference_divergence_suspicion(
        {
            "gold_available": False,
            "line_count": 50,
            "dtw_line_coverage": 0.55,
            "dtw_word_coverage": 0.40,
            "agreement_coverage_ratio": 0.10,
            "agreement_text_similarity_mean": 0.92,
            "agreement_start_p95_abs_sec": 1.0,
            "low_confidence_ratio": 0.05,
        }
    )
    assert result["suspected"] is True
    assert "no_gold_reference" in result["evidence"]


def test_infer_reference_divergence_insufficient_comparable_coverage_branch():
    module = _load_module()
    result = module._infer_reference_divergence_suspicion(
        {
            "gold_available": True,
            "line_count": 25,
            "gold_comparable_word_count": 8,
            "gold_word_coverage_ratio": 0.7,
            "gold_start_mean_abs_sec": 1.0,
            "gold_start_p95_abs_sec": 2.0,
            "dtw_line_coverage": 0.9,
            "agreement_coverage_ratio": 0.3,
            "agreement_text_similarity_mean": 0.9,
            "low_confidence_ratio": 0.05,
        }
    )
    assert result["suspected"] is False
    assert result["evidence"] == ["insufficient_comparable_coverage"]


def test_compute_lexical_line_diagnostics_rescue_and_flags():
    module = _load_module()
    line = {"index": 3}
    diag = module._compute_lexical_line_diagnostics(
        line=line,
        line_text="I can't stop stop",
        whisper_text="i cant stop",
    )
    assert diag is not None
    assert int(diag["line_token_count"]) == 4
    assert int(diag["compact_rescue"]) >= 0
    assert int(diag["apostrophe_rescue"]) >= 0
    sample = diag.get("sample")
    assert sample is None or isinstance(sample, dict)


def test_compute_lexical_line_diagnostics_none_when_no_tokens():
    module = _load_module()
    diag = module._compute_lexical_line_diagnostics(
        line={"index": 1},
        line_text="???",
        whisper_text="...",
    )
    assert diag is None
