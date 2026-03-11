#!/usr/bin/env python3
"""Analyze whether alternate timed-lyrics sources beat the current source on pre-Whisper-dominant error lines."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from y2karaoke.core.components.lyrics.lrc import parse_lrc_with_timing
from y2karaoke.core.components.lyrics.sync import fetch_from_all_sources

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "20260310T_curated_canary_with_despecha_block_guard_v2"
)
DEFAULT_GOLD_ROOT = REPO_ROOT / "benchmarks" / "gold_set_candidate" / "20260305T231015Z"
DEFAULT_MATCH = "Blinding Lights|Derniere danse|DESPECHA"


@dataclass(frozen=True)
class GoldLine:
    index: int
    text: str
    start: float


def _normalize(text: str) -> str:
    lowered = (text or "").lower()
    lowered = lowered.replace("tryna", "trying")
    lowered = lowered.replace("gon'", "gonna")
    lowered = re.sub(r"\([^)]*\)", " ", lowered)
    lowered = re.sub(r"\[[^\]]*\]", " ", lowered)
    lowered = re.sub(r"[^a-z0-9\s']+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _slug(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_timing_report"):
        stem = stem[: -len("_timing_report")]
    if "_" in stem:
        return stem.split("_", 1)[1]
    return stem


def _gold_by_slug(gold_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(gold_root.glob("*.gold.json")):
        stem = path.stem[: -len(".gold")] if path.stem.endswith(".gold") else path.stem
        if "_" in stem:
            out[stem.split("_", 1)[1]] = path
    return out


def _load_gold_lines(path: Path) -> dict[int, GoldLine]:
    data = json.loads(path.read_text(encoding="utf-8"))
    gold_lines: dict[int, GoldLine] = {}
    for line in data.get("lines", []):
        text = str(line.get("text") or "")
        index = int(line.get("line_index", 0) or 0)
        if not index or not _normalize(text):
            continue
        gold_lines[index] = GoldLine(
            index=index,
            text=text,
            start=float(line.get("start", 0.0) or 0.0),
        )
    return gold_lines


def _load_source_starts(
    title: str, artist: str
) -> dict[str, dict[int, tuple[str, float]]]:
    sources = fetch_from_all_sources(title, artist, offline=True)
    out: dict[str, dict[int, tuple[str, float]]] = {}
    for source_name, (lrc_text, _duration) in sources.items():
        if not lrc_text:
            continue
        timings = parse_lrc_with_timing(lrc_text, title, artist)
        starts: dict[int, tuple[str, float]] = {}
        for idx, (start, text) in enumerate(timings, start=1):
            norm = _normalize(text)
            if norm:
                starts[idx] = (norm, float(start))
        if starts:
            out[source_name] = starts
    return out


def _interesting_lines(
    report_lines: list[dict[str, Any]],
    gold_lines: dict[int, GoldLine],
    *,
    min_gold_error_sec: float,
    max_downstream_shift_sec: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in report_lines:
        text = str(line.get("text") or "")
        norm = _normalize(text)
        line_index = int(line.get("index", 0) or 0)
        gold = gold_lines.get(line_index)
        if gold is None:
            continue
        if _normalize(gold.text) != norm:
            continue
        start = float(line.get("start", 0.0) or 0.0)
        pre = float(line.get("pre_whisper_start", start) or start)
        final_error = abs(start - gold.start)
        if final_error < min_gold_error_sec:
            continue
        if abs(start - pre) > max_downstream_shift_sec:
            continue
        rows.append(
            {
                "line_index": line_index,
                "text": text,
                "normalized_text": norm,
                "gold_start": gold.start,
                "current_start": start,
                "pre_whisper_start": pre,
                "current_abs_error": final_error,
            }
        )
    return rows


def _format_row(row: dict[str, Any], source_rows: list[dict[str, Any]]) -> str:
    lines = [
        f"- L{row['line_index']} `{row['text']}`",
        f"  - current={row['current_start']:.3f} gold={row['gold_start']:.3f} "
        f"abs_err={row['current_abs_error']:.3f} pre_whisper={row['pre_whisper_start']:.3f}",
    ]
    for source in source_rows:
        lines.append(
            f"  - {source['source']}: start={source['start']:.3f} "
            f"abs_err={source['abs_error']:.3f} improvement={source['improvement']:+.3f}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--gold-root", type=Path, default=DEFAULT_GOLD_ROOT)
    parser.add_argument("--match", default=DEFAULT_MATCH)
    parser.add_argument("--min-gold-error-sec", type=float, default=0.5)
    parser.add_argument("--max-downstream-shift-sec", type=float, default=0.05)
    parser.add_argument("--min-improvement-sec", type=float, default=0.15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gold_map = _gold_by_slug(args.gold_root)
    pattern = re.compile(args.match, re.IGNORECASE)
    reports = sorted(args.run_dir.glob("*_timing_report.json"))
    output_rows: list[str] = []
    summary: list[dict[str, Any]] = []

    for report_path in reports:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        title = str(data.get("title") or "")
        artist = str(data.get("artist") or "")
        display = f"{artist} - {title}"
        if not pattern.search(display):
            continue
        slug = _slug(report_path)
        gold_path = gold_map.get(slug)
        if gold_path is None:
            continue
        gold_lines = _load_gold_lines(gold_path)
        source_starts = _load_source_starts(title, artist)
        rows = _interesting_lines(
            data.get("lines", []),
            gold_lines,
            min_gold_error_sec=args.min_gold_error_sec,
            max_downstream_shift_sec=args.max_downstream_shift_sec,
        )
        helpful_rows = []
        for row in rows:
            source_rows = []
            for source_name, starts in source_starts.items():
                entry = starts.get(row["line_index"])
                if entry is None:
                    continue
                alt_norm, alt_start = entry
                if alt_norm != row["normalized_text"]:
                    continue
                abs_error = abs(alt_start - row["gold_start"])
                improvement = row["current_abs_error"] - abs_error
                if improvement >= args.min_improvement_sec:
                    source_rows.append(
                        {
                            "source": source_name,
                            "start": alt_start,
                            "abs_error": abs_error,
                            "improvement": improvement,
                        }
                    )
            source_rows.sort(key=lambda item: item["abs_error"])
            if source_rows:
                helpful_rows.append((row, source_rows))

        summary.append(
            {
                "song": display,
                "pre_whisper_dominant_lines": len(rows),
                "improvable_lines": len(helpful_rows),
            }
        )
        output_rows.append(f"## {display}")
        if not helpful_rows:
            output_rows.append(
                "- No alternate source cleared the improvement threshold."
            )
            continue
        for row, source_rows in helpful_rows:
            output_rows.append(_format_row(row, source_rows))

    report_md = args.run_dir / "source_substitution_opportunities.md"
    report_json = args.run_dir / "source_substitution_opportunities.json"
    report_md.write_text("\n\n".join(output_rows) + "\n", encoding="utf-8")
    report_json.write_text(
        json.dumps({"summary": summary}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"source_substitution_opportunities: {report_md}")
    for item in summary:
        print(
            f"  {item['song']}: pre_whisper_dominant_lines={item['pre_whisper_dominant_lines']} "
            f"improvable_lines={item['improvable_lines']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
