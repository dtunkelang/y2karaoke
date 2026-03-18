"""Lexical-focused tests for benchmark suite report helpers."""

from __future__ import annotations

import importlib.util
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


def test_write_markdown_summary_includes_lexical_review_hotspots(tmp_path):
    module = _load_module()
    md_path = tmp_path / "benchmark_report.md"
    module._write_markdown_summary(
        md_path,
        run_id="run-lexical",
        manifest=Path("manifest.yaml"),
        aggregate={
            "songs_succeeded": 1,
            "songs_total": 1,
            "failed_songs": [],
            "dtw_line_coverage_mean": 0.724,
            "gold_metric_song_count": 1,
            "gold_comparable_word_count_total": 526,
            "gold_word_count_total": 526,
            "curated_canary_song_count": 1,
            "curated_canary_gold_comparable_word_count_total": 526,
            "curated_canary_gold_word_count_total": 526,
            "curated_canary_avg_abs_word_start_delta_sec_word_weighted_mean": 0.322,
            "curated_canary_gold_start_p95_abs_sec_mean": 0.72,
            "curated_canary_gold_line_duration_mean_abs_sec_mean": 0.329,
            "gold_word_coverage_ratio_total": 1.0,
            "reference_divergence_suspected_count": 0,
            "reference_divergence_suspected_ratio": 0.0,
            "quality_diagnosis_counts": {"needs_manual_review": 1},
            "lexical_review_song_count": 1,
            "lexical_hook_boundary_variant_song_count": 1,
            "lexical_hook_boundary_variant_ratio_mean": 0.214,
            "lexical_truncation_pattern_ratio_mean": 0.3689,
            "lexical_repetitive_phrase_line_ratio_mean": 0.0291,
            "timing_quality_band_counts": {"good": 1},
            "avg_abs_word_start_delta_sec_word_weighted_mean": 0.322,
            "avg_abs_word_start_delta_sec_word_weighted_mean_excluding_reference_divergence": 0.322,
            "gold_end_mean_abs_sec_mean": 0.33,
            "gold_line_duration_mean_abs_sec_mean": 0.329,
            "gold_nearest_onset_start_mean_abs_sec_mean": 0.0,
            "dtw_word_coverage_mean": 0.598,
            "dtw_phonetic_similarity_coverage_mean": 0.307,
            "low_confidence_ratio_total": 0.038,
            "agreement_coverage_ratio_total": 0.657,
            "agreement_hook_boundary_eligibility_ratio_mean": 0.714,
            "agreement_eligible_lines_total": 100,
            "agreement_matched_anchor_lines_total": 69,
            "agreement_text_similarity_mean": 0.902,
            "agreement_hook_boundary_text_similarity_mean": 0.944,
            "agreement_start_mean_abs_sec_mean": 0.38,
            "agreement_start_p95_abs_sec_mean": 0.728,
            "agreement_bad_ratio_total": 0.029,
            "agreement_severe_ratio_total": 0.009,
            "whisper_anchor_start_p95_abs_sec_mean": 0.728,
            "dtw_metric_song_count": 1,
            "line_count_total": 105,
            "dtw_metric_line_count": 105,
            "dtw_line_coverage_line_weighted_mean": 0.724,
            "timing_quality_score_line_weighted_mean": 0.761,
            "separation_cache_summary": {},
            "phase_elapsed_totals_sec": {},
            "alignment_method_counts": {},
            "lyrics_provider_counts": {},
            "lyrics_source_selection_mode_counts": {},
            "lyrics_source_routing_skip_reason_counts": {},
            "alignment_policy_hint_counts": {},
            "fallback_map_attempted_count": 0,
            "fallback_map_selected_count": 0,
            "fallback_map_rejected_count": 0,
            "fallback_map_reason_counts": {},
            "issue_tag_totals": {},
            "quality_hotspots": {
                "lexical_hook_boundary_variants": [
                    {
                        "song": "Mark Ronson - Uptown Funk",
                        "value": 0.3204,
                        "count": 33,
                    }
                ]
            },
        },
        songs=[],
    )
    markdown = md_path.read_text(encoding="utf-8")
    assert "Lexical-review hotspots" in markdown
    assert (
        "`1` song(s), hook-boundary songs `1`, hook-boundary ratio `0.214`, "
        "truncation-pattern ratio `0.369`, repetitive-phrase ratio `0.029`" in markdown
    )
    assert (
        "Hook-normalized agreement signal: eligibility `0.714`, text similarity `0.944`"
        in markdown
    )
    assert "Highest hook-boundary lexical variant ratio" in markdown
    assert "Mark Ronson - Uptown Funk: 0.3204 (count=33)" in markdown


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


def test_compute_lexical_line_diagnostics_detects_hook_boundary_variant():
    module = _load_module()
    diag = module._compute_lexical_line_diagnostics(
        line={"index": 28},
        line_text="Don't believe me just watch (come on)",
        whisper_text="Don't believe me just watch",
    )
    assert diag is not None
    assert bool(diag["hook_boundary_variant"]) is True


def test_compute_lexical_line_diagnostics_none_when_no_tokens():
    module = _load_module()
    diag = module._compute_lexical_line_diagnostics(
        line={"index": 1},
        line_text="???",
        whisper_text="...",
    )
    assert diag is None
