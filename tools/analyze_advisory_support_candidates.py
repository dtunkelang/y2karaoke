"""Rank line-level advisory support opportunities from alternate transcriptions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from tools import analyze_line_support_variants as support_tool
from y2karaoke.core.components.whisper.whisper_advisory_support import (
    advisory_candidate_bucket,
    advisory_candidate_score,
)


def _collect_candidates(
    *,
    benchmark_report: dict[str, Any],
    match_substring: str | None = None,
) -> list[dict[str, Any]]:
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    candidates: list[dict[str, Any]] = []

    for song in benchmark_report.get("songs", []):
        title = str(song.get("title") or "")
        artist = str(song.get("artist") or "")
        label = f"{artist} - {title}"
        if match_substring and match_substring.lower() not in label.lower():
            continue

        report_path = Path(song["report_path"])
        gold_path = Path(song["gold_path"])
        report = support_tool._load_report(report_path)
        gold = support_tool._load_report(gold_path)
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
        lines = support_tool._summarize_lines(
            report=report,
            default_segments=default_segments,
            default_words=default_words,
            aggressive_segments=aggressive_segments,
            aggressive_words=aggressive_words,
        )
        for line in lines:
            line_payload = asdict_no_dataclass(line)
            bucket = advisory_candidate_bucket(line)
            if bucket is None:
                continue
            candidates.append(
                {
                    "song": label,
                    "report_path": str(report_path),
                    "audio_path": audio_path,
                    "bucket": bucket,
                    "score": round(advisory_candidate_score(line), 3),
                    **line_payload,
                }
            )
    candidates.sort(
        key=lambda item: (-float(item["score"]), item["song"], item["index"])
    )
    return candidates


def asdict_no_dataclass(summary: Any) -> dict[str, Any]:
    return {
        "index": int(summary.index),
        "text": str(summary.text),
        "current_window_word_count": int(summary.current_window_word_count),
        "default_window_word_count": int(summary.default_window_word_count),
        "aggressive_window_word_count": int(summary.aggressive_window_word_count),
        "default_best_segment_text": str(summary.default_best_segment_text),
        "aggressive_best_segment_text": str(summary.aggressive_best_segment_text),
        "default_best_overlap": float(summary.default_best_overlap),
        "aggressive_best_overlap": float(summary.aggressive_best_overlap),
        "aggressive_gain": bool(summary.aggressive_gain),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_report", help="Benchmark report JSON path")
    parser.add_argument(
        "--match", help="Optional case-insensitive song substring filter"
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    benchmark_report = support_tool._load_report(Path(args.benchmark_report))
    candidates = _collect_candidates(
        benchmark_report=benchmark_report,
        match_substring=args.match,
    )
    if args.json:
        print(json.dumps({"candidates": candidates}, indent=2))
        return 0

    for candidate in candidates:
        print(
            f"{candidate['bucket']} | {candidate['song']} | "
            f"line {candidate['index']} | score={candidate['score']:.3f}"
        )
        print(f"  text: {candidate['text']}")
        print(
            "  windows current/default/aggressive: "
            f"{candidate['current_window_word_count']}/"
            f"{candidate['default_window_word_count']}/"
            f"{candidate['aggressive_window_word_count']}"
        )
        print(
            "  overlap default/aggressive: "
            f"{candidate['default_best_overlap']:.3f}/"
            f"{candidate['aggressive_best_overlap']:.3f}"
        )
        print(f"  aggressive segment: {candidate['aggressive_best_segment_text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
