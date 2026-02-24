#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run visual extraction eval across benchmark set")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/benchmark_songs.yaml"),
    )
    p.add_argument(
        "--gold-dir",
        type=Path,
        default=Path("benchmarks/gold_set_karaoke_seed"),
        help="Directory containing *.visual.gold.json benchmark files",
    )
    p.add_argument(
        "--lrc-dir",
        type=Path,
        default=Path("benchmarks/reference_lrc"),
        help="Optional directory containing per-song local *.lrc references",
    )
    p.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("benchmarks/results/visual_eval"),
    )
    p.add_argument(
        "--summary-md",
        type=Path,
        default=Path("visual_metrics_summary.md"),
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=Path("benchmarks/results/visual_eval_summary.json"),
    )
    return p.parse_args()


def _f1_or_none(song_report: dict[str, Any], key: str) -> float | None:
    block = song_report.get(key)
    if isinstance(block, dict):
        val = block.get("f1")
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def _floor3(value: float) -> float:
    return math.floor(float(value) * 1000.0) / 1000.0


def main() -> int:
    args = _parse_args()
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    songs = manifest.get("songs", []) if isinstance(manifest, dict) else []

    results_md: list[str] = []
    summary_rows_md: list[str] = []
    aggregate_rows: list[dict[str, Any]] = []

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    if args.summary_json.parent:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.lrc_dir.mkdir(parents=True, exist_ok=True)

    for idx, song in enumerate(songs, 1):
        artist = str(song["artist"])
        title = str(song["title"])
        song_label = f"{artist} - {title}"
        pattern = f"{idx:02d}_*.visual.gold.json"
        candidates = sorted(args.gold_dir.glob(pattern))

        if not candidates:
            msg = f"No visual gold file found for index {idx:02d}"
            results_md.append(f"## {song_label}\nERROR: {msg}\n")
            summary_rows_md.append(f"| {idx:02d} | {song_label} | NOT FOUND | NOT FOUND |")
            aggregate_rows.append(
                {"index": idx, "artist": artist, "title": title, "status": "not_found"}
            )
            continue

        gold_file = candidates[0]
        report_path = args.reports_dir / f"{idx:02d}.json"
        cmd = [
            sys.executable,
            "tools/evaluate_visual_lyrics_quality.py",
            "--gold-json",
            str(gold_file),
            "--title",
            title,
            "--artist",
            artist,
            "--output-json",
            str(report_path),
        ]
        lrc_candidates = sorted(args.lrc_dir.glob(f"{idx:02d}_*.lrc"))
        if lrc_candidates:
            cmd.extend(["--lrc-file", str(lrc_candidates[0])])
        else:
            # Snapshot fetched LRC references for reproducible future runs.
            slug = f"{artist}-{title}".lower()
            safe = "".join(ch if ch.isalnum() else "-" for ch in slug)
            safe = "-".join(part for part in safe.split("-") if part)
            cmd.extend(
                [
                    "--write-lrc-file",
                    str(args.lrc_dir / f"{idx:02d}_{safe}.lrc"),
                ]
            )

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            err = (res.stderr or res.stdout).strip()
            results_md.append(f"## {song_label}\nERROR: {err}\n")
            summary_rows_md.append(f"| {idx:02d} | {song_label} | ERROR | ERROR |")
            aggregate_rows.append(
                {
                    "index": idx,
                    "artist": artist,
                    "title": title,
                    "status": "error",
                    "error": err,
                    "gold_json": str(gold_file),
                }
            )
            continue

        payload = json.loads(report_path.read_text(encoding="utf-8"))
        strict = payload["strict"]
        repeat = payload["repeat_capped"]
        strict_f1 = float(strict["f1"])
        repeat_f1 = float(repeat["f1"])

        strict_line = (
            f"strict: precision={strict['precision']:.4f} "
            f"recall={strict['recall']:.4f} "
            f"f1={strict_f1:.4f} "
            f"matched={strict['matched_token_count']}/"
            f"{strict['reference_token_count']} "
            f"ext={strict['extracted_token_count']}"
        )
        repeat_line = (
            f"repeat_capped: precision={repeat['precision']:.4f} "
            f"recall={repeat['recall']:.4f} "
            f"f1={repeat_f1:.4f} "
            f"matched={repeat['matched_token_count']}/"
            f"{repeat['reference_token_count']} "
            f"ext={repeat['extracted_token_count']}"
        )
        results_md.append(f"## {song_label}\n{strict_line}\n{repeat_line}\n")
        summary_rows_md.append(
            f"| {idx:02d} | {song_label} | {strict_f1:.4f} | {repeat_f1:.4f} |"
        )
        aggregate_rows.append(
            {
                "index": idx,
                "artist": artist,
                "title": title,
                "status": "ok",
                "gold_json": str(gold_file),
                "report_json": str(report_path),
                "reference_source": payload.get("reference_source", {}),
                "strict": strict,
                "repeat_capped": repeat,
            }
        )

    strict_values = [
        v for row in aggregate_rows if row.get("status") == "ok" for v in [_f1_or_none(row, "strict")] if v is not None
    ]
    repeat_values = [
        v
        for row in aggregate_rows
        if row.get("status") == "ok"
        for v in [_f1_or_none(row, "repeat_capped")]
        if v is not None
    ]
    aggregate_summary = {
        "song_count": len(songs),
        "evaluated_count": sum(1 for r in aggregate_rows if r.get("status") == "ok"),
        "error_count": sum(1 for r in aggregate_rows if r.get("status") == "error"),
        "missing_count": sum(1 for r in aggregate_rows if r.get("status") == "not_found"),
        "strict_f1_min": min(strict_values) if strict_values else None,
        "strict_f1_p10": _pct(strict_values, 0.10),
        "strict_f1_p20": _pct(strict_values, 0.20),
        "strict_f1_mean": statistics.fmean(strict_values) if strict_values else None,
        "strict_f1_median": statistics.median(strict_values) if strict_values else None,
        "repeat_capped_f1_min": min(repeat_values) if repeat_values else None,
        "repeat_capped_f1_p10": _pct(repeat_values, 0.10),
        "repeat_capped_f1_p20": _pct(repeat_values, 0.20),
        "repeat_capped_f1_mean": statistics.fmean(repeat_values) if repeat_values else None,
        "repeat_capped_f1_median": (
            statistics.median(repeat_values) if repeat_values else None
        ),
    }
    recommendations = None
    if strict_values and repeat_values:
        # Conservative suite gates (aggregate) plus very permissive per-song floors.
        recommendations = {
            "guardrails": {
                "per_song": {
                    "min_visual_eval_strict_f1": _floor3(min(strict_values)),
                    "min_visual_eval_repeat_capped_f1": _floor3(min(repeat_values)),
                },
                "aggregate": {
                    "min_visual_eval_strict_f1_mean": _floor3(
                        float(aggregate_summary["strict_f1_mean"]) * 0.95
                    ),
                    "min_visual_eval_repeat_capped_f1_mean": _floor3(
                        float(aggregate_summary["repeat_capped_f1_mean"]) * 0.95
                    ),
                    "min_visual_eval_strict_f1_median": _floor3(
                        float(aggregate_summary["strict_f1_median"]) * 0.95
                    ),
                    "min_visual_eval_repeat_capped_f1_median": _floor3(
                        float(aggregate_summary["repeat_capped_f1_median"]) * 0.95
                    ),
                },
            }
        }

    args.summary_md.write_text(
        "# Visual Extraction Quality Summary\n\n"
        "| Index | Song | Strict F1 | Repeat-Capped F1 |\n"
        "|---|---|---|---|\n"
        + "\n".join(summary_rows_md)
        + "\n\n"
        + "\n".join(results_md),
        encoding="utf-8",
    )
    args.summary_json.write_text(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "gold_dir": str(args.gold_dir),
                "lrc_dir": str(args.lrc_dir),
                "reports_dir": str(args.reports_dir),
                "summary": aggregate_summary,
                "recommendations": recommendations,
                "songs": aggregate_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Results written to {args.summary_md}")
    print(f"Aggregate JSON written to {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
