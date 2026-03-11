#!/usr/bin/env python3
"""Classify dominant per-song alignment failure modes from a benchmark report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_report(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "benchmark_report.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _song_name(song: dict[str, Any]) -> str:
    return f"{song.get('artist', '')} - {song.get('title', '')}".strip()


def _num(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _classify_song(song: dict[str, Any]) -> dict[str, Any]:
    metrics = song.get("metrics", {}) or {}
    diagnostics = song.get("alignment_diagnostics", {}) or {}
    policy_hint = song.get("alignment_policy_hint", {}) or {}
    lexical_diag = song.get("lexical_mismatch_diagnostics", {}) or {}
    skip_reasons = metrics.get("agreement_skip_reason_counts", {}) or {}

    agreement_cov_raw = _num(metrics.get("agreement_coverage_ratio"))
    agreement_p95_raw = _num(metrics.get("agreement_start_p95_abs_sec"))
    eligibility_ratio_raw = _num(metrics.get("agreement_eligibility_ratio"))
    match_ratio_raw = _num(metrics.get("agreement_match_ratio_within_eligible"))
    line_cov_raw = _num(metrics.get("dtw_line_coverage"))

    agreement_cov = agreement_cov_raw or 0.0
    agreement_p95 = agreement_p95_raw or 0.0
    eligibility_ratio = eligibility_ratio_raw or 0.0
    match_ratio = match_ratio_raw or 0.0
    line_cov = line_cov_raw or 0.0
    low_conf = _num(metrics.get("low_confidence_ratio")) or 0.0
    gold_cov = _num(metrics.get("gold_word_coverage_ratio")) or 0.0
    gold_start_mean = _num(metrics.get("gold_start_mean_abs_sec")) or 0.0
    hint = str(policy_hint.get("hint") or "")
    trunc_ratio = _num(lexical_diag.get("truncation_pattern_ratio")) or 0.0
    repetitive_ratio = _num(lexical_diag.get("repetitive_phrase_line_ratio")) or 0.0

    low_text = int(skip_reasons.get("low_text_similarity", 0) or 0)
    low_overlap = int(skip_reasons.get("low_token_overlap", 0) or 0)
    outside_window = int(skip_reasons.get("anchor_outside_window", 0) or 0)
    issue_tags = set(diagnostics.get("issue_tags", []) or [])
    status = (song.get("status") or "").strip()

    lexical_signal = low_text + low_overlap
    repetition_signal = outside_window + (
        2 if "timing_delta_clamped" in issue_tags else 0
    )
    has_agreement_data = any(
        value is not None
        for value in (
            agreement_cov_raw,
            agreement_p95_raw,
            eligibility_ratio_raw,
            match_ratio_raw,
            line_cov_raw,
        )
    )
    strong_drift_signal = (
        line_cov >= 0.7
        and agreement_p95 >= 1.1
        and eligibility_ratio >= 0.6
        and match_ratio >= 0.45
    )
    repetition_dominant = repetition_signal >= 3 and lexical_signal <= 1
    lexical_review_dominant = (
        hint == "review_dtw_lexical_matching"
        and gold_cov >= 0.75
        and gold_start_mean <= 0.5
        and agreement_cov >= 0.4
        and agreement_p95 <= 0.9
        and low_conf <= 0.1
        and (trunc_ratio >= 0.15 or repetitive_ratio >= 0.02)
    )

    label = "mixed_or_unclear"
    evidence: list[str] = []
    if not has_agreement_data:
        evidence = ["insufficient_agreement_metrics", f"status={status or 'unknown'}"]
    elif lexical_review_dominant:
        label = "lexical_hook_variant_matching"
        evidence = [
            f"alignment_hint={hint}",
            f"truncation_ratio={trunc_ratio:.3f}",
            f"repetitive_phrase_ratio={repetitive_ratio:.3f}",
            f"agreement_p95={agreement_p95:.3f}s",
        ]
    elif repetition_dominant and match_ratio < 0.75:
        label = "repetition_handling"
        evidence = [
            f"anchor_outside_window_skips={outside_window}",
            f"timing_delta_clamped={'timing_delta_clamped' in issue_tags}",
            f"match_ratio={match_ratio:.3f}",
        ]
    elif strong_drift_signal:
        label = "timing_drift"
        evidence = [
            f"dtw_line_coverage={line_cov:.3f}",
            f"agreement_start_p95={agreement_p95:.3f}s",
            f"eligibility={eligibility_ratio:.3f}",
            f"match_ratio={match_ratio:.3f}",
        ]
    elif agreement_cov < 0.25 or eligibility_ratio < 0.35 or lexical_signal >= 2:
        label = "sparse_lexical_comparability"
        evidence = [
            f"agreement_cov={agreement_cov:.3f}",
            f"eligibility={eligibility_ratio:.3f}",
            f"low_text_or_overlap_skips={lexical_signal}",
        ]
    elif line_cov >= 0.6 and agreement_p95 > 0.9:
        label = "timing_drift"
        evidence = [
            f"dtw_line_coverage={line_cov:.3f}",
            f"agreement_start_p95={agreement_p95:.3f}s",
            f"match_ratio={match_ratio:.3f}",
        ]
    else:
        evidence = [
            f"agreement_cov={agreement_cov:.3f}",
            f"agreement_p95={agreement_p95:.3f}s",
            f"match_ratio={match_ratio:.3f}",
        ]

    return {
        "song": _song_name(song),
        "label": label,
        "evidence": evidence,
    }


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Alignment Failure Mode Classification", ""]
    lines.append("| Song | Dominant Failure Mode | Evidence |")
    lines.append("|---|---|---|")
    for row in rows:
        evidence = "; ".join(row.get("evidence", []) or [])
        lines.append(f"| {row['song']} | {row['label']} | {evidence} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Benchmark report path or run directory",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path (default: <run>/failure_mode_report.json)",
    )
    p.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Output markdown path (default: <run>/failure_mode_report.md)",
    )
    p.add_argument(
        "--match",
        type=str,
        default="",
        help="Case-insensitive substring filter on song name",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report_path = args.report.expanduser().resolve()
    report = _load_report(report_path)
    songs = report.get("songs", []) or []
    rows = [_classify_song(song) for song in songs if isinstance(song, dict)]
    if args.match.strip():
        needle = args.match.strip().lower()
        rows = [row for row in rows if needle in row["song"].lower()]

    output_root = report_path if report_path.is_dir() else report_path.parent
    out_json = args.output_json or output_root / "failure_mode_report.json"
    out_md = args.output_md or output_root / "failure_mode_report.md"
    out_json.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    _write_markdown(out_md, rows)

    print("alignment_failure_classification: OK")
    print(f"  rows={len(rows)}")
    print(f"  output_json={out_json}")
    print(f"  output_md={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
