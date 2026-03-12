"""Aggregate-only resume gold-root inheritance tests."""

from __future__ import annotations

import argparse
import importlib.util
import json
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


def _write_manifest(path: Path) -> None:
    path.write_text(
        (
            "songs:\n"
            "  - artist: Artist A\n"
            "    title: Alpha\n"
            "    youtube_id: aaaaaaaaaaa\n"
            "    youtube_url: https://www.youtube.com/watch?v=aaaaaaaaaaa\n"
        ),
        encoding="utf-8",
    )


def test_prepare_run_context_inherits_gold_root_for_aggregate_only_resume(tmp_path):
    module = _load_module()
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path)
    run_dir = tmp_path / "results" / "resume-me"
    run_dir.mkdir(parents=True)
    inherited_gold_root = tmp_path / "gold-candidate"
    (run_dir / "benchmark_progress.json").write_text(
        json.dumps({"options": {"gold_root": str(inherited_gold_root)}}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        manifest=manifest_path,
        match="",
        max_songs=0,
        output_root=tmp_path / "results",
        run_id=None,
        resume_run_dir=run_dir,
        resume_latest=False,
        aggregate_only=True,
        gold_root=module.DEFAULT_GOLD_ROOT,
        agreement_baseline_report=None,
        offline=False,
        force=False,
        strategy="hybrid_dtw",
        scenario="default",
        no_whisper_map_lrc_dtw=False,
        cache_dir=None,
        min_timing_quality_score_line_weighted=0.58,
    )

    (
        _resolved_manifest,
        _songs,
        _resolved_run_dir,
        _run_id,
        _env,
        run_signature,
        gold_root,
        _baseline_aggregate,
    ) = module._prepare_run_context(args)

    assert gold_root == inherited_gold_root.resolve()
    assert run_signature["gold_root"] == str(inherited_gold_root.resolve())


def test_prepare_run_context_keeps_explicit_gold_root_on_aggregate_only_resume(
    tmp_path,
):
    module = _load_module()
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path)
    run_dir = tmp_path / "results" / "resume-me"
    run_dir.mkdir(parents=True)
    (run_dir / "benchmark_progress.json").write_text(
        json.dumps({"options": {"gold_root": str(tmp_path / "old-gold")}}),
        encoding="utf-8",
    )
    explicit_gold_root = tmp_path / "new-gold"
    args = argparse.Namespace(
        manifest=manifest_path,
        match="",
        max_songs=0,
        output_root=tmp_path / "results",
        run_id=None,
        resume_run_dir=run_dir,
        resume_latest=False,
        aggregate_only=True,
        gold_root=explicit_gold_root,
        agreement_baseline_report=None,
        offline=False,
        force=False,
        strategy="hybrid_dtw",
        scenario="default",
        no_whisper_map_lrc_dtw=False,
        cache_dir=None,
        min_timing_quality_score_line_weighted=0.58,
    )

    (
        _resolved_manifest,
        _songs,
        _resolved_run_dir,
        _run_id,
        _env,
        run_signature,
        gold_root,
        _baseline_aggregate,
    ) = module._prepare_run_context(args)

    assert gold_root == explicit_gold_root.resolve()
    assert run_signature["gold_root"] == str(explicit_gold_root.resolve())


def test_prepare_run_context_infers_gold_root_from_cached_results_when_report_stale(
    tmp_path,
):
    module = _load_module()
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path)
    run_dir = tmp_path / "results" / "resume-me"
    run_dir.mkdir(parents=True)
    stale_gold_root = tmp_path / "stale-gold"
    inherited_gold_root = tmp_path / "gold-candidate"
    (run_dir / "benchmark_progress.json").write_text(
        json.dumps({"options": {"gold_root": str(stale_gold_root)}}),
        encoding="utf-8",
    )
    (run_dir / "01_artist-a-alpha_result.json").write_text(
        json.dumps(
            {
                "artist": "Artist A",
                "title": "Alpha",
                "status": "ok",
                "run_signature": {"gold_root": str(inherited_gold_root)},
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        manifest=manifest_path,
        match="",
        max_songs=0,
        output_root=tmp_path / "results",
        run_id=None,
        resume_run_dir=run_dir,
        resume_latest=False,
        aggregate_only=True,
        gold_root=module.DEFAULT_GOLD_ROOT,
        agreement_baseline_report=None,
        offline=False,
        force=False,
        strategy="hybrid_dtw",
        scenario="default",
        no_whisper_map_lrc_dtw=False,
        cache_dir=None,
        min_timing_quality_score_line_weighted=0.58,
    )

    (
        _resolved_manifest,
        _songs,
        _resolved_run_dir,
        _run_id,
        _env,
        run_signature,
        gold_root,
        _baseline_aggregate,
    ) = module._prepare_run_context(args)

    assert gold_root == inherited_gold_root.resolve()
    assert run_signature["gold_root"] == str(inherited_gold_root.resolve())
