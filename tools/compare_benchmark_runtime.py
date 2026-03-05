#!/usr/bin/env python3
"""Compare runtime metrics between two benchmark reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _resolve_report(path: Path) -> Path:
    return path / "benchmark_report.json" if path.is_dir() else path


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(_resolve_report(path).read_text(encoding="utf-8"))


def _song_name(song: dict[str, Any]) -> str:
    return f"{song.get('artist', '')} - {song.get('title', '')}".strip()


def _phase_elapsed(song: dict[str, Any], phase: str) -> float | None:
    phase_map = song.get("phase_durations_sec", {}) or {}
    if not isinstance(phase_map, dict):
        return None
    value = phase_map.get(phase)
    return float(value) if isinstance(value, (int, float)) else None


def _build_song_index(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for song in doc.get("songs", []) or []:
        if not isinstance(song, dict):
            continue
        out[_song_name(song)] = song
    return out


def _aggregate_elapsed_total(doc: dict[str, Any]) -> float:
    aggregate = doc.get("aggregate", {}) or {}
    if not isinstance(aggregate, dict):
        return 0.0
    value_total = aggregate.get("sum_song_elapsed_total_sec")
    if isinstance(value_total, (int, float)):
        return float(value_total)
    value_executed = aggregate.get("sum_song_elapsed_sec")
    if isinstance(value_executed, (int, float)):
        return float(value_executed)
    return 0.0


def _compare_reports(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    baseline_agg = baseline.get("aggregate", {}) or {}
    candidate_agg = candidate.get("aggregate", {}) or {}
    baseline_songs = _build_song_index(baseline)
    candidate_songs = _build_song_index(candidate)
    common = sorted(set(baseline_songs) & set(candidate_songs))

    rows: list[dict[str, Any]] = []
    alignment_delta_comparable = 0
    whisper_delta_comparable = 0
    for name in common:
        b = baseline_songs[name]
        c = candidate_songs[name]
        b_elapsed = float(b.get("elapsed_sec", 0.0) or 0.0)
        c_elapsed = float(c.get("elapsed_sec", 0.0) or 0.0)
        b_alignment = _phase_elapsed(b, "alignment")
        c_alignment = _phase_elapsed(c, "alignment")
        b_whisper = _phase_elapsed(b, "whisper")
        c_whisper = _phase_elapsed(c, "whisper")
        alignment_delta = (
            round(c_alignment - b_alignment, 3)
            if b_alignment is not None and c_alignment is not None
            else None
        )
        whisper_delta = (
            round(c_whisper - b_whisper, 3)
            if b_whisper is not None and c_whisper is not None
            else None
        )
        if alignment_delta is not None:
            alignment_delta_comparable += 1
        if whisper_delta is not None:
            whisper_delta_comparable += 1
        rows.append(
            {
                "song": name,
                "elapsed_baseline": round(b_elapsed, 3),
                "elapsed_candidate": round(c_elapsed, 3),
                "elapsed_delta_sec": round(c_elapsed - b_elapsed, 3),
                "alignment_delta_sec": alignment_delta,
                "whisper_delta_sec": whisper_delta,
                "fallback_attempted_baseline": int(
                    (b.get("metrics", {}) or {}).get("fallback_map_attempted", 0) or 0
                ),
                "fallback_attempted_candidate": int(
                    (c.get("metrics", {}) or {}).get("fallback_map_attempted", 0) or 0
                ),
            }
        )

    rows.sort(key=lambda row: float(row.get("elapsed_delta_sec", 0.0)), reverse=True)
    suite_wall_delta = round(
        float(candidate.get("suite_wall_elapsed_sec", 0.0) or 0.0)
        - float(baseline.get("suite_wall_elapsed_sec", 0.0) or 0.0),
        3,
    )
    sum_song_executed_delta = round(
        float(candidate_agg.get("sum_song_elapsed_sec", 0.0) or 0.0)
        - float(baseline_agg.get("sum_song_elapsed_sec", 0.0) or 0.0),
        3,
    )
    sum_song_delta = round(
        _aggregate_elapsed_total(candidate) - _aggregate_elapsed_total(baseline),
        3,
    )
    warnings: list[str] = []
    if len(common) > 0 and alignment_delta_comparable < len(common):
        warnings.append(
            "Alignment phase deltas are partially non-comparable across reports."
        )
    if len(common) > 0 and whisper_delta_comparable < len(common):
        warnings.append(
            "Whisper phase deltas are partially non-comparable across reports."
        )
    if abs(sum_song_delta - sum_song_executed_delta) >= 1.0:
        warnings.append(
            "Executed vs total elapsed deltas diverge; one report may be aggregate-only."
        )
    return {
        "common_song_count": len(common),
        "alignment_delta_comparable_song_count": alignment_delta_comparable,
        "whisper_delta_comparable_song_count": whisper_delta_comparable,
        "suite_wall_elapsed_delta_sec": suite_wall_delta,
        "sum_song_elapsed_total_delta_sec": sum_song_delta,
        "sum_song_elapsed_executed_delta_sec": sum_song_executed_delta,
        "warnings": warnings,
        "rows": rows,
    }


def _format_delta(value: Any) -> str:
    return f"{float(value):+.3f}" if isinstance(value, (int, float)) else "n/a"


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Benchmark Runtime Delta", ""]
    lines.append(f"- Common songs: `{payload.get('common_song_count', 0)}`")
    lines.append(
        f"- Rows emitted: `{payload.get('rows_emitted', len(payload.get('rows', []) or []))}`"
    )
    filters = payload.get("filters", {}) or {}
    lines.append(
        f"- Filters: `top={int(filters.get('top', 0) or 0)}`, "
        f"`only_positive_delta={bool(filters.get('only_positive_delta', False))}`"
    )
    lines.append(
        "- Comparable phase deltas:"
        f" `alignment={int(payload.get('alignment_delta_comparable_song_count', 0) or 0)}`"
        f", `whisper={int(payload.get('whisper_delta_comparable_song_count', 0) or 0)}`"
    )
    lines.append(
        f"- Suite wall elapsed delta: `{float(payload.get('suite_wall_elapsed_delta_sec', 0.0) or 0.0):+.3f}s`"
    )
    lines.append(
        "- Sum song elapsed delta: "
        f"`total={float(payload.get('sum_song_elapsed_total_delta_sec', 0.0) or 0.0):+.3f}s`, "
        f"`executed={float(payload.get('sum_song_elapsed_executed_delta_sec', 0.0) or 0.0):+.3f}s`"
    )
    warn_rows = payload.get("warnings", []) or []
    if warn_rows:
        lines.append("- Warnings:")
        for warning in warn_rows:
            lines.append(f"  - {warning}")
    lines.append("")
    lines.append(
        "| Song | Elapsed Δ (s) | Alignment Δ (s) | Whisper Δ (s) | Fallback (B->C) |"
    )
    lines.append("|---|---:|---:|---:|---|")
    for row in payload.get("rows", []) or []:
        lines.append(
            f"| {row.get('song', '')} | {float(row.get('elapsed_delta_sec', 0.0)):+.3f} | "
            f"{_format_delta(row.get('alignment_delta_sec'))} | "
            f"{_format_delta(row.get('whisper_delta_sec'))} | "
            f"{int(row.get('fallback_attempted_baseline', 0))}->{int(row.get('fallback_attempted_candidate', 0))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, required=True, help="Baseline report path or run dir"
    )
    parser.add_argument(
        "--candidate", type=Path, required=True, help="Candidate report path or run dir"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Limit output rows to top-N by elapsed delta (0 keeps all)",
    )
    parser.add_argument(
        "--only-positive-delta",
        action="store_true",
        help="Keep only songs with positive elapsed delta",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None, help="Output JSON path"
    )
    parser.add_argument(
        "--output-md", type=Path, default=None, help="Output markdown path"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_path = args.baseline.expanduser().resolve()
    candidate_path = args.candidate.expanduser().resolve()
    baseline_doc = _load_report(baseline_path)
    candidate_doc = _load_report(candidate_path)
    payload = _compare_reports(baseline_doc, candidate_doc)
    rows = payload.get("rows", []) or []
    if args.only_positive_delta:
        rows = [
            row for row in rows if float(row.get("elapsed_delta_sec", 0.0) or 0.0) > 0.0
        ]
    if args.top > 0:
        rows = rows[: args.top]
    payload["rows"] = rows
    payload["rows_emitted"] = len(rows)
    payload["filters"] = {
        "top": int(args.top),
        "only_positive_delta": bool(args.only_positive_delta),
    }

    out_root = candidate_path if candidate_path.is_dir() else candidate_path.parent
    out_json = args.output_json or (out_root / "runtime_delta.json")
    out_md = args.output_md or (out_root / "runtime_delta.md")
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown(out_md, payload)
    print("benchmark_runtime_delta: OK")
    print(f"  common_songs={payload.get('common_song_count', 0)}")
    print(f"  output_json={out_json}")
    print(f"  output_md={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
