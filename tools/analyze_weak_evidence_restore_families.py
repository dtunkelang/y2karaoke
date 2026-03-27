#!/usr/bin/env python3
"""Classify weak-evidence restore rows into narrower line families."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import analyze_weak_evidence_restore_decisions as restore_tool  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z]+", "", text.lower())


def _line_tokens(text: str) -> set[str]:
    return {
        token
        for token in (_normalize_token(part) for part in text.split())
        if token and token != "vocal"
    }


def _neighbor_overlap_tokens(
    report_lines: dict[int, dict[str, Any]],
    line_index: int,
) -> tuple[int, int]:
    current = _line_tokens(str(report_lines[line_index].get("text", "")))
    prev_tokens = _line_tokens(
        str(report_lines.get(line_index - 1, {}).get("text", ""))
    )
    next_tokens = _line_tokens(
        str(report_lines.get(line_index + 1, {}).get("text", ""))
    )
    return len(current & prev_tokens), len(current & next_tokens)


def _classify_family(
    *,
    reason: str,
    word_count: int,
    suffix_support_tokens: int,
    prev_overlap_tokens: int,
    next_overlap_tokens: int,
) -> str:
    if reason != "suffix_only_support":
        return reason
    overlap = max(prev_overlap_tokens, next_overlap_tokens)
    if word_count <= 3 and suffix_support_tokens >= 2 and overlap >= 2:
        return "repeated_short_hook_suffix_support"
    return "sparse_tail_suffix_support"


def analyze(
    stage_trace: dict[str, Any], timing_report: dict[str, Any]
) -> dict[str, Any]:
    base = restore_tool.analyze(stage_trace, timing_report)
    report_lines = {int(line["index"]): line for line in timing_report.get("lines", [])}
    rows: list[dict[str, Any]] = []
    for row in base.get("rows", []):
        line_index = int(row["line_index"])
        report_line = report_lines.get(line_index, {})
        word_count = len(report_line.get("words") or [])
        prev_overlap_tokens, next_overlap_tokens = _neighbor_overlap_tokens(
            report_lines,
            line_index,
        )
        family = _classify_family(
            reason=str(row["reason"]),
            word_count=word_count,
            suffix_support_tokens=int(row.get("suffix_support_tokens", 0)),
            prev_overlap_tokens=prev_overlap_tokens,
            next_overlap_tokens=next_overlap_tokens,
        )
        enriched = dict(row)
        enriched.update(
            {
                "word_count": word_count,
                "prev_overlap_tokens": prev_overlap_tokens,
                "next_overlap_tokens": next_overlap_tokens,
                "family": family,
            }
        )
        rows.append(enriched)
    return {
        "stage": base.get("stage"),
        "before_stage": base.get("before_stage"),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_trace_json", help="Whisper stage trace JSON")
    parser.add_argument("timing_report_json", help="Timing report JSON")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    payload = analyze(
        _load_json(Path(args.stage_trace_json)),
        _load_json(Path(args.timing_report_json)),
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    for row in payload["rows"]:
        print(
            f"line {row['line_index']}: {row['family']} "
            f"(reason={row['reason']}, prev_overlap={row['prev_overlap_tokens']}, "
            f"next_overlap={row['next_overlap_tokens']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
