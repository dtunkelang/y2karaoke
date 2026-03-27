"""Rank line-level override opportunities across alternate support families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from tools import analyze_fuzzy_segment_support_variants as fuzzy_tool
from tools import analyze_line_support_variants as support_tool
from y2karaoke.core.components.whisper.whisper_advisory_support import (
    advisory_candidate_bucket,
    advisory_candidate_score,
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _classify_opportunity(
    *,
    current_window_word_count: int,
    advisory_bucket: str | None,
    fuzzy_joint_score: float,
    fuzzy_estimated_start: float | None,
    current_start: float,
) -> str:
    if advisory_bucket is not None:
        return "advisory_exact_start"
    if current_window_word_count == 0 and fuzzy_joint_score >= 0.95:
        return "merged_aggressive_only"
    if (
        fuzzy_joint_score >= 0.6
        and fuzzy_estimated_start is not None
        and abs(fuzzy_estimated_start - current_start) <= 0.45
    ):
        return "fuzzy_span_candidate"
    if current_window_word_count == 0:
        return "zero_support"
    return "no_clear_override"


def _estimate_error(value: float | None, gold: float | None) -> float | None:
    if value is None or gold is None:
        return None
    return abs(value - gold)


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _line_gold_map(gold: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for idx, line in enumerate(gold.get("lines", []), start=1):
        out[idx] = line
    return out


def _analyze_song(
    *,
    song: dict[str, Any],
    model: WhisperModel,
) -> list[dict[str, Any]]:
    report_path = Path(song["report_path"])
    gold_path = Path(song["gold_path"])
    report = _load_json(report_path)
    gold = _load_json(gold_path)
    gold_by_index = _line_gold_map(gold)
    audio_path = str(Path(gold["audio_path"]).expanduser().resolve())
    default_segments, default_words = support_tool._transcribe_variant(
        model=model,
        audio_path=audio_path,
        aggressive=False,
    )
    aggressive_segments, aggressive_words = support_tool._transcribe_variant(
        model=model,
        audio_path=audio_path,
        aggressive=True,
    )
    support_lines = support_tool._summarize_lines(
        report=report,
        default_segments=default_segments,
        default_words=default_words,
        aggressive_segments=aggressive_segments,
        aggressive_words=aggressive_words,
    )
    support_by_index = {int(line.index): line for line in support_lines}
    fuzzy_lines = fuzzy_tool._analyze_report(
        report=report,
        default_segments=default_segments,
        aggressive_segments=aggressive_segments,
        language="es" if "con calma" in str(song.get("title", "")).lower() else "en",
        line_indexes=set(),
    )
    fuzzy_by_index = {int(line["index"]): line for line in fuzzy_lines}

    results: list[dict[str, Any]] = []
    label = f"{song.get('artist', '')} - {song.get('title', '')}".strip(" -")
    for report_line in report.get("lines", []):
        idx = int(report_line["index"])
        support = support_by_index[idx]
        fuzzy = fuzzy_by_index[idx]["aggressive_best"]
        gold_line = gold_by_index.get(idx)
        gold_start = (
            float(gold_line["start"])
            if gold_line is not None and "start" in gold_line
            else None
        )
        advisory_bucket = advisory_candidate_bucket(support)
        current_start = float(report_line["start"])
        pre_start = (
            float(report_line["pre_whisper_start"])
            if isinstance(report_line.get("pre_whisper_start"), (int, float))
            else None
        )
        fuzzy_start = (
            float(fuzzy["estimated_span_start"])
            if fuzzy.get("estimated_span_start") is not None
            else None
        )
        opportunity = _classify_opportunity(
            current_window_word_count=int(
                report_line.get("whisper_window_word_count", 0)
            ),
            advisory_bucket=advisory_bucket,
            fuzzy_joint_score=float(fuzzy["joint_score"]),
            fuzzy_estimated_start=fuzzy_start,
            current_start=current_start,
        )
        results.append(
            {
                "song": label,
                "index": idx,
                "text": str(report_line["text"]),
                "opportunity": opportunity,
                "advisory_bucket": advisory_bucket,
                "advisory_score": round(advisory_candidate_score(support), 3),
                "current_window_word_count": int(
                    report_line.get("whisper_window_word_count", 0)
                ),
                "aggressive_overlap": float(support.aggressive_best_overlap),
                "fuzzy_joint_score": float(fuzzy["joint_score"]),
                "fuzzy_estimated_start": fuzzy_start,
                "current_start": current_start,
                "pre_start": pre_start,
                "gold_start": gold_start,
                "current_start_error": _estimate_error(current_start, gold_start),
                "pre_start_error": _estimate_error(pre_start, gold_start),
                "fuzzy_start_error": _estimate_error(fuzzy_start, gold_start),
            }
        )
    return results


def _rank_key(item: dict[str, Any]) -> tuple[float, float, str, int]:
    current_err = float(item["current_start_error"] or 0.0)
    potential_gain = current_err - float(item["fuzzy_start_error"] or current_err)
    if item["advisory_bucket"] is not None:
        potential_gain = max(
            potential_gain,
            current_err - float(item["pre_start_error"] or current_err),
        )
    return (
        -potential_gain,
        -current_err,
        str(item["song"]),
        int(item["index"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_report", help="Benchmark report JSON path")
    parser.add_argument("--match", help="Optional song substring filter")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    report = _load_json(Path(args.benchmark_report))
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    rows: list[dict[str, Any]] = []
    for song in report.get("songs", []):
        label = f"{song.get('artist', '')} - {song.get('title', '')}".strip(" -")
        if args.match and args.match.lower() not in label.lower():
            continue
        rows.extend(_analyze_song(song=song, model=model))
    rows.sort(key=_rank_key)

    if args.json:
        print(json.dumps({"rows": rows}, indent=2))
        return 0

    for row in rows:
        print(
            f"{row['opportunity']} | {row['song']} | line {row['index']} | "
            f"err={_fmt_float(row['current_start_error'])}"
        )
        print(f"  text: {row['text']}")
        print(
            f"  current/pre/fuzzy/gold: {_fmt_float(row['current_start'])} / "
            f"{_fmt_float(row['pre_start'])} / "
            f"{_fmt_float(row['fuzzy_estimated_start'])} / "
            f"{_fmt_float(row['gold_start'])}"
        )
        print(
            f"  advisory={row['advisory_bucket']} score={row['advisory_score']:.3f} "
            f"aggressive_overlap={row['aggressive_overlap']:.3f} "
            f"fuzzy_joint={row['fuzzy_joint_score']:.3f} "
            f"window_words={row['current_window_word_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
