#!/usr/bin/env python3
"""Estimate benchmark-level agreement metrics under the guarded two-layer policy."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

pack_tool = cast(Any, importlib.import_module("tools.analyze_two_layer_agreement_pack"))


def _resolve_report(path: Path) -> Path:
    return path / "benchmark_report.json" if path.is_dir() else path


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(_resolve_report(path).read_text(encoding="utf-8"))


def _safe_int(raw: Any) -> int:
    return int(raw or 0)


def _safe_float(raw: Any) -> float:
    return float(raw or 0.0)


def analyze(
    report_doc: dict[str, Any],
    *,
    min_text_similarity: float = 0.58,
    min_token_overlap: float = 0.5,
    min_line_words: int = 6,
    min_anchor_surplus_words: int = 15,
    min_anchor_words: int = 20,
) -> dict[str, Any]:
    aggregate = report_doc.get("aggregate", {}) or {}
    if not isinstance(aggregate, dict):
        aggregate = {}

    pack_payload = pack_tool.analyze(
        report_doc,
        min_text_similarity=min_text_similarity,
        min_token_overlap=min_token_overlap,
        min_line_words=min_line_words,
        min_anchor_surplus_words=min_anchor_surplus_words,
        min_anchor_words=min_anchor_words,
    )

    baseline_line_count = _safe_int(aggregate.get("line_count_total"))
    baseline_count = _safe_int(aggregate.get("agreement_count_total"))
    baseline_eligible = _safe_int(aggregate.get("agreement_eligible_lines_total"))
    baseline_matched = _safe_int(aggregate.get("agreement_matched_anchor_lines_total"))
    baseline_good = _safe_int(aggregate.get("agreement_good_lines_total"))
    baseline_bad = _safe_int(aggregate.get("agreement_bad_lines_total"))
    baseline_warn = _safe_int(aggregate.get("agreement_warn_lines_total"))
    baseline_severe = _safe_int(aggregate.get("agreement_severe_lines_total"))
    baseline_comparability_report = aggregate.get("agreement_comparability_report", [])
    if not isinstance(baseline_comparability_report, list):
        baseline_comparability_report = []

    eligible_gain = _safe_int(
        pack_payload.get("adjusted_eligible_lines_total")
    ) - _safe_int(pack_payload.get("baseline_eligible_lines_total"))
    matched_gain = _safe_int(
        pack_payload.get("adjusted_matched_lines_total")
    ) - _safe_int(pack_payload.get("baseline_matched_lines_total"))

    adjusted_eligible = baseline_eligible + eligible_gain
    adjusted_matched = baseline_matched + matched_gain
    adjusted_count = baseline_count + matched_gain
    adjusted_good = baseline_good + matched_gain

    baseline_report_by_song: dict[str, dict[str, Any]] = {}
    for raw_row in baseline_comparability_report:
        if not isinstance(raw_row, dict):
            continue
        song_name = str(raw_row.get("song", "")).strip()
        if song_name:
            baseline_report_by_song[song_name] = raw_row

    rows = []
    for row in pack_payload.get("rows", []) or []:
        if not isinstance(row, dict):
            continue
        recovered = _safe_int(row.get("recovered_lines"))
        if recovered <= 0:
            continue
        baseline_row_eligible = _safe_int(row.get("baseline_eligible_lines"))
        baseline_row_matched = _safe_int(row.get("baseline_matched_lines"))
        adjusted_row_eligible = _safe_int(row.get("adjusted_eligible_lines"))
        adjusted_row_matched = _safe_int(row.get("adjusted_matched_lines"))
        rows.append(
            {
                "song": str(row.get("song", "")),
                "recovered_lines": recovered,
                "baseline_eligible_lines": baseline_row_eligible,
                "baseline_matched_lines": baseline_row_matched,
                "adjusted_eligible_lines": adjusted_row_eligible,
                "adjusted_matched_lines": adjusted_row_matched,
                "baseline_coverage_ratio": _safe_float(
                    row.get("baseline_coverage_ratio")
                ),
                "adjusted_coverage_ratio": _safe_float(
                    row.get("adjusted_coverage_ratio")
                ),
                "baseline_fraction": (
                    f"{baseline_row_matched}/{baseline_row_eligible}"
                    if baseline_row_eligible
                    else "0/0"
                ),
                "adjusted_fraction": (
                    f"{adjusted_row_matched}/{adjusted_row_eligible}"
                    if adjusted_row_eligible
                    else "0/0"
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            -_safe_int(row.get("recovered_lines")),
            str(row.get("song", "")),
        )
    )
    adjusted_coverage_ratio = (
        adjusted_matched / adjusted_eligible if adjusted_eligible else 0.0
    )
    adjusted_bad_ratio = (
        baseline_bad / baseline_line_count if baseline_line_count else 0.0
    )
    adjusted_good_ratio = (
        adjusted_good / baseline_line_count if baseline_line_count else 0.0
    )
    adjusted_warn_ratio = (
        baseline_warn / baseline_line_count if baseline_line_count else 0.0
    )
    adjusted_severe_ratio = (
        baseline_severe / baseline_line_count if baseline_line_count else 0.0
    )
    baseline_hotspots: list[dict[str, Any]] = []
    prototype_hotspots: list[dict[str, Any]] = []
    all_song_names = {
        *baseline_report_by_song.keys(),
        *(str(row.get("song", "")).strip() for row in rows),
    }
    for song_name in sorted(name for name in all_song_names if name):
        baseline_row = baseline_report_by_song.get(song_name, {})
        prototype_row = next(
            (row for row in rows if str(row.get("song", "")).strip() == song_name),
            {},
        )
        baseline_song_eligible = _safe_int(baseline_row.get("eligible_lines"))
        baseline_song_matched = _safe_int(baseline_row.get("matched_lines_anchor"))
        adjusted_song_eligible = _safe_int(
            prototype_row.get("adjusted_eligible_lines", baseline_song_eligible)
        )
        adjusted_song_matched = _safe_int(
            prototype_row.get("adjusted_matched_lines", baseline_song_matched)
        )
        baseline_hotspots.append(
            {
                "song": song_name,
                "eligible_lines": baseline_song_eligible,
                "matched_lines": baseline_song_matched,
                "match_ratio_within_eligible": (
                    baseline_song_matched / baseline_song_eligible
                    if baseline_song_eligible
                    else 0.0
                ),
            }
        )
        prototype_hotspots.append(
            {
                "song": song_name,
                "eligible_lines": adjusted_song_eligible,
                "matched_lines": adjusted_song_matched,
                "match_ratio_within_eligible": (
                    adjusted_song_matched / adjusted_song_eligible
                    if adjusted_song_eligible
                    else 0.0
                ),
            }
        )

    baseline_hotspots.sort(
        key=lambda row: (
            _safe_float(row.get("match_ratio_within_eligible")),
            _safe_int(row.get("eligible_lines")),
            str(row.get("song", "")),
        )
    )
    prototype_hotspots.sort(
        key=lambda row: (
            _safe_float(row.get("match_ratio_within_eligible")),
            _safe_int(row.get("eligible_lines")),
            str(row.get("song", "")),
        )
    )
    return {
        "guard": {
            "min_line_words": min_line_words,
            "min_anchor_surplus_words": min_anchor_surplus_words,
            "min_anchor_words": min_anchor_words,
            "min_text_similarity": min_text_similarity,
            "min_token_overlap": min_token_overlap,
        },
        "assumptions": {
            "recovered_lines_count_as_good_matches": True,
            "baseline_bad_warn_severe_line_counts_unchanged": True,
        },
        "baseline": {
            "line_count_total": baseline_line_count,
            "agreement_count_total": baseline_count,
            "agreement_eligible_lines_total": baseline_eligible,
            "agreement_matched_anchor_lines_total": baseline_matched,
            "agreement_good_lines_total": baseline_good,
            "agreement_bad_lines_total": baseline_bad,
            "agreement_warn_lines_total": baseline_warn,
            "agreement_severe_lines_total": baseline_severe,
            "agreement_coverage_ratio_total": _safe_float(
                aggregate.get("agreement_coverage_ratio_total")
            ),
            "agreement_bad_ratio_total": _safe_float(
                aggregate.get("agreement_bad_ratio_total")
            ),
            "agreement_good_ratio_total": _safe_float(
                aggregate.get("agreement_good_ratio_total")
            ),
            "agreement_warn_ratio_total": _safe_float(
                aggregate.get("agreement_warn_ratio_total")
            ),
            "agreement_severe_ratio_total": _safe_float(
                aggregate.get("agreement_severe_ratio_total")
            ),
        },
        "prototype": {
            "line_count_total": baseline_line_count,
            "agreement_count_total": adjusted_count,
            "agreement_eligible_lines_total": adjusted_eligible,
            "agreement_matched_anchor_lines_total": adjusted_matched,
            "agreement_good_lines_total": adjusted_good,
            "agreement_bad_lines_total": baseline_bad,
            "agreement_warn_lines_total": baseline_warn,
            "agreement_severe_lines_total": baseline_severe,
            "agreement_coverage_ratio_total": round(adjusted_coverage_ratio, 4),
            "agreement_bad_ratio_total": round(adjusted_bad_ratio, 4),
            "agreement_good_ratio_total": round(adjusted_good_ratio, 4),
            "agreement_warn_ratio_total": round(adjusted_warn_ratio, 4),
            "agreement_severe_ratio_total": round(adjusted_severe_ratio, 4),
        },
        "delta": {
            "agreement_count_total": matched_gain,
            "agreement_eligible_lines_total": eligible_gain,
            "agreement_matched_anchor_lines_total": matched_gain,
            "agreement_good_lines_total": matched_gain,
            "agreement_coverage_ratio_total": round(
                adjusted_coverage_ratio
                - _safe_float(aggregate.get("agreement_coverage_ratio_total")),
                4,
            ),
            "agreement_bad_ratio_total": round(
                adjusted_bad_ratio
                - _safe_float(aggregate.get("agreement_bad_ratio_total")),
                4,
            ),
            "agreement_good_ratio_total": round(
                adjusted_good_ratio
                - _safe_float(aggregate.get("agreement_good_ratio_total")),
                4,
            ),
        },
        "recovered_song_count": len(rows),
        "baseline_hotspots": baseline_hotspots,
        "prototype_hotspots": prototype_hotspots,
        "rows": rows,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    baseline = payload.get("baseline", {}) or {}
    prototype = payload.get("prototype", {}) or {}
    delta = payload.get("delta", {}) or {}
    guard = payload.get("guard", {}) or {}
    lines = ["# Two-Layer Benchmark Prototype", ""]
    lines.append(
        "- Guard: "
        f"`line_words>={int(guard.get('min_line_words', 0) or 0)}`, "
        f"`anchor_surplus>={int(guard.get('min_anchor_surplus_words', 0) or 0)}`, "
        f"`anchor_words>={int(guard.get('min_anchor_words', 0) or 0)}`"
    )
    lines.append("- Assumption: recovered lines count as good matches.")
    lines.append("- Bad/warn/severe counts stay fixed.")
    lines.append("")
    lines.append("| Metric | Baseline | Prototype | Delta |")
    lines.append("|---|---:|---:|---:|")
    for key in [
        "agreement_eligible_lines_total",
        "agreement_matched_anchor_lines_total",
        "agreement_good_lines_total",
        "agreement_bad_lines_total",
        "agreement_coverage_ratio_total",
        "agreement_bad_ratio_total",
        "agreement_good_ratio_total",
    ]:
        lines.append(
            f"| {key} | {baseline.get(key, 0)} | "
            f"{prototype.get(key, 0)} | {delta.get(key, 0)} |"
        )
    lines.append("")
    lines.append("## Hotspot Order")
    lines.append("")
    lines.append("| Baseline Rank | Prototype Rank | Song | Baseline | Prototype |")
    lines.append("|---:|---:|---|---:|---:|")
    baseline_hotspots = payload.get("baseline_hotspots", []) or []
    prototype_hotspots = payload.get("prototype_hotspots", []) or []
    prototype_rank = {
        str(row.get("song", "")): index + 1
        for index, row in enumerate(prototype_hotspots)
        if isinstance(row, dict)
    }
    for index, row in enumerate(baseline_hotspots):
        if not isinstance(row, dict):
            continue
        song_name = str(row.get("song", ""))
        prototype_row = next(
            (
                candidate
                for candidate in prototype_hotspots
                if isinstance(candidate, dict)
                and str(candidate.get("song", "")) == song_name
            ),
            {},
        )
        lines.append(
            f"| {index + 1} | {prototype_rank.get(song_name, '-')} | {song_name} | "
            f"{_safe_int(row.get('matched_lines'))}/"
            f"{_safe_int(row.get('eligible_lines'))} | "
            f"{_safe_int(prototype_row.get('matched_lines'))}/"
            f"{_safe_int(prototype_row.get('eligible_lines'))} |"
        )
    lines.append("")
    lines.append("| Song | Recovered | Baseline | Prototype |")
    lines.append("|---|---:|---:|---:|")
    for row in payload.get("rows", []) or []:
        lines.append(
            f"| {row.get('song', '')} | {int(row.get('recovered_lines', 0) or 0)} | "
            f"{row.get('baseline_fraction', '0/0')} | "
            f"{row.get('adjusted_fraction', '0/0')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="Benchmark report JSON or run directory")
    parser.add_argument("--json", action="store_true", help="Emit JSON to stdout")
    parser.add_argument(
        "--output-json", type=Path, default=None, help="Output JSON path"
    )
    parser.add_argument(
        "--output-md", type=Path, default=None, help="Output markdown path"
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    payload = analyze(_load_report(report_path))
    resolved = _resolve_report(report_path)
    output_json = args.output_json or (
        resolved.parent / "two_layer_benchmark_prototype.json"
    )
    output_md = args.output_md or (resolved.parent / "two_layer_benchmark_prototype.md")
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("two_layer_benchmark_prototype: OK")
        print(f"  output_json={output_json}")
        print(f"  output_md={output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
