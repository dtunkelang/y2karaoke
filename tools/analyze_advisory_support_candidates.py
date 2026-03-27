"""Rank line-level advisory support opportunities from alternate transcriptions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from tools import analyze_line_support_variants as support_tool


def _candidate_score(line: dict[str, Any]) -> float:
    return (
        float(line["aggressive_best_overlap"]) * 10.0
        + min(int(line["aggressive_window_word_count"]), 8) * 0.2
        - min(int(line["current_window_word_count"]), 8) * 0.15
    )


def _candidate_bucket(line: dict[str, Any]) -> str | None:
    overlap = float(line["aggressive_best_overlap"])
    current_words = int(line["current_window_word_count"])
    default_words = int(line["default_window_word_count"])
    aggressive_words = int(line["aggressive_window_word_count"])
    if overlap >= 0.95 and aggressive_words >= 3 and default_words == 0:
        return "high_confidence"
    if overlap >= 0.75 and aggressive_words >= 3 and current_words <= 3:
        return "medium_confidence"
    return None


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
            bucket = _candidate_bucket(line_payload)
            if bucket is None:
                continue
            candidates.append(
                {
                    "song": label,
                    "report_path": str(report_path),
                    "audio_path": audio_path,
                    "bucket": bucket,
                    "score": round(_candidate_score(line_payload), 3),
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
