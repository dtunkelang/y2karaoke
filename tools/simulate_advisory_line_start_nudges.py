"""Simulate start-only advisory nudges from aggressive support candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import analyze_advisory_support_candidates as candidate_tool
from tools import analyze_line_support_variants as support_tool


def _load_gold(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _simulate_song(
    *,
    song: dict[str, Any],
    candidate_lines: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    report = support_tool._load_report(Path(song["report_path"]))
    gold = _load_gold(Path(song["gold_path"]))
    gold_lines = {idx + 1: line for idx, line in enumerate(gold.get("lines", []))}

    current_start_errors: list[float] = []
    simulated_start_errors: list[float] = []
    current_end_errors: list[float] = []
    simulated_lines: list[dict[str, Any]] = []

    for line in report.get("lines", []):
        index = int(line["index"])
        gold_line = gold_lines.get(index)
        if gold_line is None:
            continue
        current_start = float(line["start"])
        current_end = float(line["end"])
        simulated_start = current_start
        candidate = candidate_lines.get(index)
        if candidate is not None:
            aggressive_segment = candidate.get("aggressive_best_segment_text", "")
            if aggressive_segment:
                # Recompute exact aggressive segment start from the already-ranked candidate payload.
                simulated_start = float(candidate.get("advisory_start", current_start))
        current_start_errors.append(abs(current_start - float(gold_line["start"])))
        simulated_start_errors.append(abs(simulated_start - float(gold_line["start"])))
        current_end_errors.append(abs(current_end - float(gold_line["end"])))
        simulated_lines.append(
            {
                "index": index,
                "text": line["text"],
                "current_start": current_start,
                "simulated_start": simulated_start,
                "gold_start": float(gold_line["start"]),
                "current_start_error": abs(current_start - float(gold_line["start"])),
                "simulated_start_error": abs(
                    simulated_start - float(gold_line["start"])
                ),
                "end": current_end,
                "gold_end": float(gold_line["end"]),
                "end_error": abs(current_end - float(gold_line["end"])),
                "candidate": candidate is not None,
            }
        )

    return {
        "song": f"{song['artist']} - {song['title']}",
        "current_start_mean": sum(current_start_errors)
        / max(1, len(current_start_errors)),
        "simulated_start_mean": sum(simulated_start_errors)
        / max(1, len(simulated_start_errors)),
        "current_end_mean": sum(current_end_errors) / max(1, len(current_end_errors)),
        "lines": simulated_lines,
    }


def _collect_song_candidates(
    benchmark_report: dict[str, Any], match_substring: str | None
) -> tuple[list[dict[str, Any]], dict[str, dict[int, dict[str, Any]]]]:
    raw_candidates = candidate_tool._collect_candidates(
        benchmark_report=benchmark_report,
        match_substring=match_substring,
    )
    by_song: dict[str, dict[int, dict[str, Any]]] = {}

    for song in benchmark_report.get("songs", []):
        title = str(song.get("title") or "")
        artist = str(song.get("artist") or "")
        label = f"{artist} - {title}"
        if match_substring and match_substring.lower() not in label.lower():
            continue
        report = support_tool._load_report(Path(song["report_path"]))
        gold = _load_gold(Path(song["gold_path"]))
        audio_path = str(Path(gold["audio_path"]).expanduser().resolve())
        model = candidate_tool.WhisperModel(
            "large-v3", device="cpu", compute_type="int8"
        )
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
        summaries = support_tool._summarize_lines(
            report=report,
            default_segments=default_segments,
            default_words=default_words,
            aggressive_segments=aggressive_segments,
            aggressive_words=aggressive_words,
        )
        candidate_map = {
            int(c["index"]): dict(c) for c in raw_candidates if c["song"] == label
        }
        for summary in summaries:
            idx = int(summary.index)
            if idx not in candidate_map:
                continue
            for seg in aggressive_segments:
                if seg.text == summary.aggressive_best_segment_text:
                    candidate_map[idx]["advisory_start"] = float(seg.start)
                    break
        by_song[label] = candidate_map
    return raw_candidates, by_song


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_report", help="Benchmark report JSON path")
    parser.add_argument(
        "--match", help="Optional case-insensitive song substring filter"
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    benchmark_report = support_tool._load_report(Path(args.benchmark_report))
    _raw_candidates, candidates_by_song = _collect_song_candidates(
        benchmark_report,
        args.match,
    )
    results = []
    for song in benchmark_report.get("songs", []):
        label = f"{song['artist']} - {song['title']}"
        if args.match and args.match.lower() not in label.lower():
            continue
        results.append(
            _simulate_song(song=song, candidate_lines=candidates_by_song.get(label, {}))
        )

    if args.json:
        print(json.dumps({"songs": results}, indent=2))
        return 0

    for result in results:
        print(result["song"])
        print(
            f"  start mean: {result['current_start_mean']:.3f} -> {result['simulated_start_mean']:.3f}"
        )
        print(f"  end mean: {result['current_end_mean']:.3f}")
        for line in result["lines"]:
            if not line["candidate"]:
                continue
            print(
                f"  line {line['index']}: {line['current_start_error']:.3f} -> "
                f"{line['simulated_start_error']:.3f} ({line['text']})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
