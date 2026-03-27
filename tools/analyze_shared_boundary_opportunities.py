"""Analyze adjacent line pairs for shared-boundary repair opportunities."""

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


def _normalize(text: str) -> str:
    return "".join(_TOKEN_RE.findall(text.lower()))


def _tokenize(text: str) -> list[str]:
    return [_normalize(token) for token in text.split() if _normalize(token)]


def _line_last_parenthetical_token(line: dict[str, Any]) -> str:
    words = line.get("words") or []
    if not words:
        return ""
    last_text = str(words[-1].get("text", ""))
    if ")" not in last_text:
        return ""
    tokens = _tokenize(last_text)
    for token in reversed(tokens):
        if len(token) >= 2:
            return token
    return tokens[-1] if tokens else ""


def _next_prefix_target(
    report_line: dict[str, Any], *, prefix_count: int = 3
) -> float | None:
    line_tokens = _tokenize(str(report_line.get("text", "")))[:prefix_count]
    if len(line_tokens) < max(2, prefix_count):
        return None
    whisper_words = report_line.get("whisper_window_words") or []
    normalized_words = [_normalize(str(word.get("text", ""))) for word in whisper_words]
    for idx in range(0, len(normalized_words) - len(line_tokens) + 1):
        candidate = normalized_words[idx : idx + len(line_tokens)]
        if candidate != line_tokens:
            continue
        return float(whisper_words[idx]["start"])
    return None


def _parenthetical_tail_support_end(
    report_line: dict[str, Any],
    *,
    search_end: float,
) -> float | None:
    tail_token = _line_last_parenthetical_token(report_line)
    if not tail_token:
        return None
    support_end: float | None = None
    for word in report_line.get("whisper_window_words") or []:
        start = float(word["start"])
        end = float(word["end"])
        if start < float(report_line["end"]) or end > search_end:
            continue
        if _normalize(str(word.get("text", ""))) != tail_token:
            continue
        support_end = end
    return support_end


def _analyze_pair(
    prev_line: dict[str, Any],
    next_line: dict[str, Any],
    gold_prev: dict[str, Any],
    gold_next: dict[str, Any],
) -> dict[str, Any] | None:
    target_start = _next_prefix_target(next_line)
    if target_start is None:
        return None
    tail_support_end = _parenthetical_tail_support_end(
        prev_line, search_end=target_start
    )
    current_next_start = float(next_line["start"])
    current_prev_end = float(prev_line["end"])
    if tail_support_end is None and target_start >= current_next_start - 0.15:
        return None
    return {
        "prev_index": int(prev_line["index"]),
        "prev_text": str(prev_line["text"]),
        "next_index": int(next_line["index"]),
        "next_text": str(next_line["text"]),
        "current_prev_end": current_prev_end,
        "gold_prev_end": float(gold_prev["end"]),
        "current_next_start": current_next_start,
        "gold_next_start": float(gold_next["start"]),
        "suggested_prev_end": (
            min(target_start - 0.05, tail_support_end + 0.1)
            if tail_support_end is not None
            else None
        ),
        "suggested_next_start": target_start,
        "tail_support_end": tail_support_end,
        "next_prefix_target": target_start,
        "next_start_improvement": round(current_next_start - target_start, 3),
        "prev_end_extension": (
            round(
                min(target_start - 0.05, tail_support_end + 0.1) - current_prev_end, 3
            )
            if tail_support_end is not None
            else None
        ),
        "family": (
            "parenthetical_followup_boundary"
            if tail_support_end is not None
            else "next_prefix_only"
        ),
    }


def analyze(report: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    report_lines = report.get("lines", [])
    gold_lines = gold.get("lines", [])
    for prev_line, next_line, gold_prev, gold_next in zip(
        report_lines,
        report_lines[1:],
        gold_lines,
        gold_lines[1:],
    ):
        row = _analyze_pair(prev_line, next_line, gold_prev, gold_next)
        if row is None:
            continue
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -(row["next_start_improvement"] or 0.0),
            -(row["prev_end_extension"] or 0.0),
            row["prev_index"],
        )
    )
    return {
        "title": report.get("title"),
        "artist": report.get("artist"),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timing_report", help="Timing report JSON path")
    parser.add_argument("gold_json", help="Gold timing JSON path")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    payload = analyze(
        _load_json(Path(args.timing_report)),
        _load_json(Path(args.gold_json)),
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    label = " - ".join(
        part for part in [payload.get("artist"), payload.get("title")] if part
    )
    print(label)
    for row in payload["rows"]:
        print(
            f"lines {row['prev_index']}/{row['next_index']} {row['family']}: "
            f"next {row['current_next_start']:.3f}->{row['suggested_next_start']:.3f}"
        )
        if row["suggested_prev_end"] is not None:
            print(
                "  prev end "
                f"{row['current_prev_end']:.3f}->{row['suggested_prev_end']:.3f}"
            )
        print(f"  {row['prev_text']} || {row['next_text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
