#!/usr/bin/env python3
"""Recommend prioritized human-guided correction tasks from benchmark output."""

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


def _build_actions(song: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    agreement_cov = float(metrics.get("agreement_coverage_ratio", 0.0) or 0.0)
    agreement_p95 = float(metrics.get("agreement_start_p95_abs_sec", 0.0) or 0.0)
    low_conf = float(metrics.get("low_confidence_ratio", 0.0) or 0.0)
    dtw_line = float(metrics.get("dtw_line_coverage", 0.0) or 0.0)
    fallback_attempted = int(metrics.get("fallback_map_attempted", 0) or 0)
    fallback_selected = int(metrics.get("fallback_map_selected", 0) or 0)

    if agreement_cov < 0.35:
        actions.append(
            "Use gold editor jump-to-anchor + snap-to-onset to label comparable lines faster."
        )
    if agreement_p95 > 0.9:
        actions.append(
            "Micro-nudge line/word timings with hotkeys; focus on first word onset per line."
        )
    if low_conf > 0.08:
        actions.append(
            "Review low-confidence sections first; add word-level corrections around unclear syllables."
        )
    if dtw_line < 0.9:
        actions.append(
            "Inspect timing-source quality (LRC drift/offset) and anchor key phrase starts manually."
        )
    if fallback_attempted > 0 and fallback_selected == 0:
        actions.append(
            "Check fallback-map rejection reason in diagnostics before broad timing edits."
        )
    if not actions:
        actions.append(
            "No urgent manual correction needed; keep as spot-check candidate."
        )
    return actions


def _score_song(metrics: dict[str, Any]) -> float:
    agreement_cov = float(metrics.get("agreement_coverage_ratio", 0.0) or 0.0)
    agreement_p95 = float(metrics.get("agreement_start_p95_abs_sec", 0.0) or 0.0)
    low_conf = float(metrics.get("low_confidence_ratio", 0.0) or 0.0)
    dtw_line = float(metrics.get("dtw_line_coverage", 0.0) or 0.0)
    fallback_attempted = float(metrics.get("fallback_map_attempted", 0.0) or 0.0)
    fallback_selected = float(metrics.get("fallback_map_selected", 0.0) or 0.0)
    fallback_penalty = max(0.0, fallback_attempted - fallback_selected)
    # Higher score = better candidate for human correction priority.
    return (
        (max(0.0, 0.4 - agreement_cov) * 2.5)
        + (max(0.0, agreement_p95 - 0.8) * 1.6)
        + (low_conf * 1.4)
        + (max(0.0, 0.9 - dtw_line) * 1.2)
        + (fallback_penalty * 0.08)
    )


def _recommend(doc: dict[str, Any], top: int) -> dict[str, Any]:
    songs = doc.get("songs", []) or []
    rows: list[dict[str, Any]] = []
    for song in songs:
        if not isinstance(song, dict):
            continue
        metrics_raw = song.get("metrics", {}) or {}
        metrics = metrics_raw if isinstance(metrics_raw, dict) else {}
        score = _score_song(metrics)
        rows.append(
            {
                "song": _song_name(song),
                "status": str(song.get("status", "")),
                "priority_score": round(score, 3),
                "agreement_coverage_ratio": round(
                    float(metrics.get("agreement_coverage_ratio", 0.0) or 0.0), 3
                ),
                "agreement_start_p95_abs_sec": round(
                    float(metrics.get("agreement_start_p95_abs_sec", 0.0) or 0.0), 3
                ),
                "low_confidence_ratio": round(
                    float(metrics.get("low_confidence_ratio", 0.0) or 0.0), 3
                ),
                "dtw_line_coverage": round(
                    float(metrics.get("dtw_line_coverage", 0.0) or 0.0), 3
                ),
                "actions": _build_actions(song, metrics),
            }
        )
    rows = [row for row in rows if row.get("status") == "ok"]
    rows.sort(key=lambda row: float(row.get("priority_score", 0.0)), reverse=True)
    if top > 0:
        rows = rows[:top]
    return {
        "song_count_considered": len(rows),
        "top": top,
        "rows": rows,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Human Guidance Task Recommendations", ""]
    lines.append(f"- Songs considered: `{payload.get('song_count_considered', 0)}`")
    lines.append(f"- Top limit: `{payload.get('top', 0)}`")
    lines.append("")
    lines.append(
        "| Song | Priority | Agreement Cov | Agreement P95 (s) | Low Conf | DTW Line Cov |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in payload.get("rows", []) or []:
        lines.append(
            f"| {row.get('song', '')} | {float(row.get('priority_score', 0.0)):.3f} | "
            f"{float(row.get('agreement_coverage_ratio', 0.0)):.3f} | "
            f"{float(row.get('agreement_start_p95_abs_sec', 0.0)):.3f} | "
            f"{float(row.get('low_confidence_ratio', 0.0)):.3f} | "
            f"{float(row.get('dtw_line_coverage', 0.0)):.3f} |"
        )
        actions = row.get("actions", []) or []
        for action in actions:
            lines.append(f"- {row.get('song', '')}: {action}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, required=True, help="Run dir or report path"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="How many songs to include (0 means all)",
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
    report_path = args.report.expanduser().resolve()
    report = _load_report(report_path)
    payload = _recommend(report, top=int(args.top))

    out_root = report_path if report_path.is_dir() else report_path.parent
    out_json = args.output_json or (out_root / "human_guidance_tasks.json")
    out_md = args.output_md or (out_root / "human_guidance_tasks.md")
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown(out_md, payload)
    print("human_guidance_tasks: OK")
    print(f"  rows={len(payload.get('rows', []))}")
    print(f"  output_json={out_json}")
    print(f"  output_md={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
