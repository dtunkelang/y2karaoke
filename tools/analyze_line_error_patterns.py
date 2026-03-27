"""Summarize line-level timing failure patterns from a timing report."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[a-z0-9']+")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _line_gold_map(gold: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for idx, line in enumerate(gold.get("lines", []), start=1):
        out[idx] = line
    return out


def _contaminated_successors(
    contamination: dict[str, Any], contaminated_classes: set[str]
) -> set[int]:
    out: set[int] = set()
    for gap in contamination.get("gaps", []):
        if gap.get("classification") in contaminated_classes:
            gap_index = gap.get("gap_index")
            if isinstance(gap_index, int):
                out.add(gap_index + 1)
    return out


def _token_overlap_ratio(line_text: str, window_words: list[dict[str, Any]]) -> float:
    line_tokens = set(_tokenize(line_text))
    window_tokens = {
        token for word in window_words for token in _tokenize(str(word.get("text", "")))
    }
    if not line_tokens:
        return 0.0
    return len(line_tokens & window_tokens) / len(line_tokens)


def _estimate_error(value: float | None, gold: float | None) -> float | None:
    if value is None or gold is None:
        return None
    return abs(value - gold)


def _classify_line(
    *,
    report_line: dict[str, Any],
    gold_line: dict[str, Any] | None,
    contaminated_successors: set[int],
) -> dict[str, Any]:
    idx = int(report_line["index"])
    current_start = float(report_line["start"])
    current_end = float(report_line["end"])
    pre_start_raw = report_line.get("pre_whisper_start")
    pre_start = (
        float(pre_start_raw) if isinstance(pre_start_raw, (int, float)) else None
    )
    gold_start = (
        float(gold_line["start"])
        if gold_line is not None and isinstance(gold_line.get("start"), (int, float))
        else None
    )
    gold_end = (
        float(gold_line["end"])
        if gold_line is not None and isinstance(gold_line.get("end"), (int, float))
        else None
    )
    window_words = list(report_line.get("whisper_window_words", []))
    word_count = int(report_line.get("whisper_window_word_count", 0))
    overlap = _token_overlap_ratio(str(report_line["text"]), window_words)
    tags: list[str] = []
    start_error = _estimate_error(current_start, gold_start)
    end_error = _estimate_error(current_end, gold_end)
    pre_error = _estimate_error(pre_start, gold_start)

    if start_error is not None and start_error >= 0.25:
        tags.append("large_start_error")
    if end_error is not None and end_error >= 0.25:
        tags.append("large_end_error")
    if word_count == 0:
        tags.append("zero_window_support")
    elif word_count <= 2:
        tags.append("sparse_window_support")
    if overlap < 0.5:
        tags.append("low_lexical_overlap")
    elif overlap >= 0.9:
        tags.append("high_lexical_overlap")
    if idx in contaminated_successors:
        tags.append("contaminated_gap_predecessor")
    if pre_error is not None and start_error is not None:
        if pre_error + 0.05 < start_error:
            tags.append("pre_whisper_closer_than_current")
        elif start_error + 0.05 < pre_error:
            tags.append("current_closer_than_pre_whisper")
    if (
        pre_start is not None
        and current_start - pre_start >= 0.6
        and overlap >= 0.9
        and word_count >= 3
    ):
        tags.append("late_exact_window_alignment")
    if "..." in str(report_line["text"]):
        tags.append("truncated_line")

    return {
        "index": idx,
        "text": str(report_line["text"]),
        "start": current_start,
        "end": current_end,
        "gold_start": gold_start,
        "gold_end": gold_end,
        "pre_whisper_start": pre_start,
        "start_error": start_error,
        "end_error": end_error,
        "whisper_window_word_count": word_count,
        "lexical_overlap_ratio": round(overlap, 3),
        "tags": tags,
    }


def _analyze_report(
    *,
    report: dict[str, Any],
    gold: dict[str, Any],
    contamination: dict[str, Any] | None,
) -> dict[str, Any]:
    gold_by_index = _line_gold_map(gold)
    contaminated_successors = _contaminated_successors(
        contamination or {},
        {"echo_fragment", "hallucinated_interstitial", "unclear_interstitial"},
    )
    rows = [
        _classify_line(
            report_line=line,
            gold_line=gold_by_index.get(int(line["index"])),
            contaminated_successors=contaminated_successors,
        )
        for line in report.get("lines", [])
    ]
    rows.sort(
        key=lambda row: (
            -(row["start_error"] or 0.0),
            -(row["end_error"] or 0.0),
            int(row["index"]),
        )
    )
    tag_counts = Counter(
        tag
        for row in rows
        if (row["start_error"] or 0.0) >= 0.15 or (row["end_error"] or 0.0) >= 0.15
        for tag in row["tags"]
    )
    return {
        "title": report.get("title"),
        "artist": report.get("artist"),
        "alignment_method": report.get("alignment_method"),
        "rows": rows,
        "tag_counts": dict(tag_counts),
    }


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timing_report", help="Timing report JSON path")
    parser.add_argument("gold_json", help="Gold JSON path")
    parser.add_argument("--contamination-json", help="Optional contamination JSON path")
    parser.add_argument("--top", type=int, default=8, help="Rows to print")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    report = _load_json(Path(args.timing_report))
    gold = _load_json(Path(args.gold_json))
    contamination = (
        _load_json(Path(args.contamination_json)) if args.contamination_json else None
    )
    result = _analyze_report(report=report, gold=gold, contamination=contamination)

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    label = " - ".join(
        part for part in [result.get("artist"), result.get("title")] if part
    )
    print(label)
    print(f"alignment_method: {result.get('alignment_method')}")
    print(f"top_tags: {result['tag_counts']}")
    for row in result["rows"][: args.top]:
        print(
            f"line {row['index']} | start_err={_fmt(row['start_error'])} "
            f"end_err={_fmt(row['end_error'])} | {row['text']}"
        )
        print(
            f"  start/pre/gold: {_fmt(row['start'])} / "
            f"{_fmt(row['pre_whisper_start'])} / {_fmt(row['gold_start'])}"
        )
        print(
            f"  window_words={row['whisper_window_word_count']} "
            f"overlap={row['lexical_overlap_ratio']:.3f} tags={', '.join(row['tags'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
