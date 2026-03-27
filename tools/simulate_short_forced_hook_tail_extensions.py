"""Simulate end-only extensions for short hook-like forced lines."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _normalize_token(text: str) -> str:
    return "".join(_TOKEN_RE.findall(text.lower()))


def _song_label(song: dict[str, Any]) -> str:
    return f"{song.get('artist', '')} - {song.get('title', '')}".strip(" -")


def _estimate_error(current: float | None, gold: float | None) -> float | None:
    if current is None or gold is None:
        return None
    return abs(float(current) - float(gold))


def _last_token(line: dict[str, Any]) -> str:
    words = line.get("words") or []
    if not words:
        return ""
    return _normalize_token(words[-1].get("text", ""))


def _has_repeated_tail_support(
    line: dict[str, Any],
    *,
    min_repeat_gap_sec: float,
) -> bool:
    last_token = _last_token(line)
    if not last_token:
        return False
    current_end = float(line["end"])
    for word in line.get("whisper_window_words", []):
        token = _normalize_token(word.get("text", ""))
        if token != last_token:
            continue
        if float(word["start"]) < current_end + min_repeat_gap_sec:
            continue
        return True
    return False


def _candidate_target_end(
    line: dict[str, Any],
    *,
    max_word_count: int,
    min_extension_sec: float,
    max_extension_sec: float,
    min_repeat_gap_sec: float,
) -> float | None:
    words = line.get("words") or []
    if not words or len(words) > max_word_count:
        return None
    current_end = float(line["end"])
    pre_end = float(line.get("pre_whisper_end", current_end))
    extension = pre_end - current_end
    if extension < min_extension_sec or extension > max_extension_sec:
        return None
    if not _has_repeated_tail_support(line, min_repeat_gap_sec=min_repeat_gap_sec):
        return None
    return pre_end


def _simulate_song(
    report: dict[str, Any],
    gold: dict[str, Any],
    *,
    max_word_count: int,
    min_extension_sec: float,
    max_extension_sec: float,
    min_repeat_gap_sec: float,
) -> dict[str, Any]:
    current_end_errors: list[float] = []
    simulated_end_errors: list[float] = []
    lines: list[dict[str, Any]] = []
    report_lines = report.get("lines", [])
    gold_lines = gold.get("lines", [])
    for idx, (line, gold_line) in enumerate(zip(report_lines, gold_lines), start=1):
        current_end = float(line["end"])
        gold_end = float(gold_line["end"])
        simulated_end = current_end
        target_end = _candidate_target_end(
            line,
            max_word_count=max_word_count,
            min_extension_sec=min_extension_sec,
            max_extension_sec=max_extension_sec,
            min_repeat_gap_sec=min_repeat_gap_sec,
        )
        if target_end is not None:
            simulated_end = target_end
        current_err = _estimate_error(current_end, gold_end)
        simulated_err = _estimate_error(simulated_end, gold_end)
        if current_err is not None:
            current_end_errors.append(current_err)
        if simulated_err is not None:
            simulated_end_errors.append(simulated_err)
        lines.append(
            {
                "index": idx,
                "text": line.get("text", ""),
                "current_end": current_end,
                "simulated_end": simulated_end,
                "gold_end": gold_end,
                "current_end_error": current_err,
                "simulated_end_error": simulated_err,
                "candidate": target_end is not None,
            }
        )
    return {
        "current_end_mean": sum(current_end_errors) / max(1, len(current_end_errors)),
        "simulated_end_mean": sum(simulated_end_errors)
        / max(1, len(simulated_end_errors)),
        "lines": lines,
    }


def _analyze_song(
    song: dict[str, Any],
    *,
    max_word_count: int,
    min_extension_sec: float,
    max_extension_sec: float,
    min_repeat_gap_sec: float,
) -> dict[str, Any]:
    report = _load_json(Path(song["report_path"]))
    gold = _load_json(Path(song["gold_path"]))
    simulated = _simulate_song(
        report,
        gold,
        max_word_count=max_word_count,
        min_extension_sec=min_extension_sec,
        max_extension_sec=max_extension_sec,
        min_repeat_gap_sec=min_repeat_gap_sec,
    )
    return {
        "song": _song_label(song),
        "current_end_mean": simulated["current_end_mean"],
        "simulated_end_mean": simulated["simulated_end_mean"],
        "improvement": simulated["current_end_mean"] - simulated["simulated_end_mean"],
        "changed_lines": [
            line
            for line in simulated["lines"]
            if line["current_end"] != line["simulated_end"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_report", help="Benchmark report JSON path")
    parser.add_argument("--match", help="Optional song substring filter")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    parser.add_argument("--max-word-count", type=int, default=3)
    parser.add_argument("--min-extension-sec", type=float, default=0.2)
    parser.add_argument("--max-extension-sec", type=float, default=1.6)
    parser.add_argument("--min-repeat-gap-sec", type=float, default=0.03)
    args = parser.parse_args()

    report = _load_json(Path(args.benchmark_report))
    rows: list[dict[str, Any]] = []
    for song in report.get("songs", []):
        label = _song_label(song)
        if args.match and args.match.lower() not in label.lower():
            continue
        rows.append(
            _analyze_song(
                song,
                max_word_count=args.max_word_count,
                min_extension_sec=args.min_extension_sec,
                max_extension_sec=args.max_extension_sec,
                min_repeat_gap_sec=args.min_repeat_gap_sec,
            )
        )
    rows.sort(key=lambda row: (-float(row["improvement"]), row["song"]))

    if args.json:
        print(json.dumps({"songs": rows}, indent=2))
        return 0

    for row in rows:
        print(
            f"{row['song']}: {row['current_end_mean']:.3f} -> "
            f"{row['simulated_end_mean']:.3f} "
            f"(improvement {row['improvement']:.3f})"
        )
        for line in row["changed_lines"]:
            print(
                f"  line {line['index']}: {line['current_end_error']:.3f} -> "
                f"{line['simulated_end_error']:.3f} ({line['text']})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
