"""Compare timing drift across repeated lyric lines within one song."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

PAREN_RE = re.compile(r"\([^)]*\)")
TOKEN_RE = re.compile(r"[a-z0-9']+")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _normalize_text(text: str) -> str:
    text = PAREN_RE.sub("", text.lower())
    return " ".join(TOKEN_RE.findall(text))


def _gold_by_index(gold: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {idx: line for idx, line in enumerate(gold.get("lines", []), start=1)}


def _analyze(
    *,
    report: dict[str, Any],
    gold: dict[str, Any],
    min_group_size: int = 2,
) -> list[dict[str, Any]]:
    gold_map = _gold_by_index(gold)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in report.get("lines", []):
        idx = int(line["index"])
        gold_line = gold_map.get(idx)
        if gold_line is None:
            continue
        grouped[_normalize_text(str(line["text"]))].append(
            {
                "index": idx,
                "text": str(line["text"]),
                "start": float(line["start"]),
                "end": float(line["end"]),
                "pre_whisper_start": float(line["pre_whisper_start"]),
                "gold_start": float(gold_line["start"]),
                "gold_end": float(gold_line["end"]),
                "start_error": float(line["start"]) - float(gold_line["start"]),
                "end_error": float(line["end"]) - float(gold_line["end"]),
            }
        )

    results: list[dict[str, Any]] = []
    for normalized, rows in grouped.items():
        if len(rows) < min_group_size:
            continue
        rows.sort(key=lambda row: int(row["index"]))
        start_errors = [row["start_error"] for row in rows]
        end_errors = [row["end_error"] for row in rows]
        results.append(
            {
                "normalized_text": normalized,
                "occurrences": rows,
                "start_error_span": max(start_errors) - min(start_errors),
                "end_error_span": max(end_errors) - min(end_errors),
            }
        )
    results.sort(
        key=lambda item: (
            -item["start_error_span"],
            -item["end_error_span"],
            item["normalized_text"],
        )
    )
    return results


def _fmt(value: float) -> str:
    return f"{value:+.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timing_report", help="Timing report JSON path")
    parser.add_argument("gold_json", help="Gold JSON path")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    result = _analyze(
        report=_load_json(Path(args.timing_report)),
        gold=_load_json(Path(args.gold_json)),
    )

    if args.json:
        print(json.dumps({"groups": result}, indent=2))
        return 0

    for group in result:
        print(
            f"{group['normalized_text']} | "
            f"start_span={group['start_error_span']:.3f} "
            f"end_span={group['end_error_span']:.3f}"
        )
        for row in group["occurrences"]:
            print(
                f"  line {row['index']}: start_err={_fmt(row['start_error'])} "
                f"end_err={_fmt(row['end_error'])} | {row['text']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
