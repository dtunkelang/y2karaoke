#!/usr/bin/env python3
"""Profile slow songs in a benchmark report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_report(path: Path) -> dict[str, Any]:
    report_path = path / "benchmark_report.json" if path.is_dir() else path
    return json.loads(report_path.read_text(encoding="utf-8"))


def _song_name(song: dict[str, Any]) -> str:
    return f"{song.get('artist', '')} - {song.get('title', '')}".strip()


def _parse_rows(report: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    songs = report.get("songs", []) or []
    rows: list[dict[str, Any]] = []
    total_elapsed = 0.0
    for song in songs:
        if not isinstance(song, dict):
            continue
        elapsed_raw = song.get("elapsed_sec")
        elapsed = float(elapsed_raw) if isinstance(elapsed_raw, (int, float)) else 0.0
        total_elapsed += elapsed
        phase_map_raw = song.get("phase_durations_sec", {}) or {}
        phase_map = (
            {
                str(key): float(value)
                for key, value in phase_map_raw.items()
                if isinstance(key, str) and isinstance(value, (int, float))
            }
            if isinstance(phase_map_raw, dict)
            else {}
        )
        bottleneck_phase = (
            max(phase_map, key=lambda key: phase_map[key]) if phase_map else ""
        )
        metrics = song.get("metrics", {}) or {}
        rows.append(
            {
                "song": _song_name(song),
                "elapsed_sec": round(elapsed, 3),
                "status": str(song.get("status", "")),
                "last_stage_hint": str(song.get("last_stage_hint", "")),
                "cache_decisions": song.get("cache_decisions", {}) or {},
                "phase_durations_sec": {k: round(v, 2) for k, v in phase_map.items()},
                "bottleneck_phase": bottleneck_phase,
                "bottleneck_phase_share": round(
                    (
                        (phase_map.get(bottleneck_phase, 0.0) / elapsed)
                        if elapsed > 0.0 and bottleneck_phase
                        else 0.0
                    ),
                    4,
                ),
                "fallback_map_attempted": (
                    int(metrics.get("fallback_map_attempted", 0) or 0)
                    if isinstance(metrics, dict)
                    else 0
                ),
                "fallback_map_selected": (
                    int(metrics.get("fallback_map_selected", 0) or 0)
                    if isinstance(metrics, dict)
                    else 0
                ),
                "local_transcribe_cache_hits": (
                    int(metrics.get("local_transcribe_cache_hits", 0) or 0)
                    if isinstance(metrics, dict)
                    else 0
                ),
                "local_transcribe_cache_misses": (
                    int(metrics.get("local_transcribe_cache_misses", 0) or 0)
                    if isinstance(metrics, dict)
                    else 0
                ),
            }
        )
    rows.sort(key=lambda row: row["elapsed_sec"], reverse=True)
    return rows, total_elapsed


def _write_markdown(
    path: Path, rows: list[dict[str, Any]], total_elapsed: float
) -> None:
    lines = ["# Benchmark Runtime Profile", ""]
    lines.append(f"- Total elapsed across songs: `{total_elapsed:.2f}s`")
    lines.append("")
    lines.append(
        "| Song | Elapsed (s) | Share | Bottleneck | Bottleneck Share | Fallback | Local Tx Cache | Status | Last Stage |"
    )
    lines.append("|---|---:|---:|---|---:|---|---|---|---|")
    for row in rows:
        elapsed = float(row.get("elapsed_sec") or 0.0)
        share = (elapsed / total_elapsed) if total_elapsed > 0.0 else 0.0
        fallback = (
            f"{int(row.get('fallback_map_attempted', 0) or 0)}/"
            f"{int(row.get('fallback_map_selected', 0) or 0)}"
        )
        local_cache = (
            f"{int(row.get('local_transcribe_cache_hits', 0) or 0)}/"
            f"{int(row.get('local_transcribe_cache_misses', 0) or 0)}"
        )
        lines.append(
            f"| {row.get('song', '')} | {elapsed:.2f} | {share:.1%} | "
            f"{row.get('bottleneck_phase', '')} | "
            f"{float(row.get('bottleneck_phase_share', 0.0) or 0.0):.1%} | "
            f"{fallback} | {local_cache} | "
            f"{row.get('status', '')} | {row.get('last_stage_hint', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Benchmark report path or run directory",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of slowest songs to include",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path (default: <run>/runtime_profile.json)",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Output markdown path (default: <run>/runtime_profile.md)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = args.report.expanduser().resolve()
    report = _load_report(report_path)
    rows, total_elapsed = _parse_rows(report)
    if args.top > 0:
        rows = rows[: args.top]

    output_root = report_path if report_path.is_dir() else report_path.parent
    out_json = args.output_json or output_root / "runtime_profile.json"
    out_md = args.output_md or output_root / "runtime_profile.md"

    payload = {
        "rows": rows,
        "total_elapsed_sec": round(total_elapsed, 3),
        "top": int(args.top),
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown(out_md, rows, total_elapsed)

    print("benchmark_runtime_profile: OK")
    print(f"  rows={len(rows)}")
    print(f"  output_json={out_json}")
    print(f"  output_md={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
