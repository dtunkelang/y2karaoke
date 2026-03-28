#!/usr/bin/env python3
"""Find adjacent alternating 3-word hook pairs in curated gold clips."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_tokens(text: str) -> list[str]:
    return [token for token in re.sub(r"[^a-z]+", " ", text.lower()).split() if token]


def _is_alternating_pair(left: list[str], right: list[str]) -> bool:
    return (
        len(left) == 3
        and len(right) == 3
        and left[0] == right[0]
        and left[1] == right[2]
        and left[2] == right[1]
        and left[1] != left[2]
    )


def analyze(*, gold_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(gold_root.glob("*.gold.json")):
        data = _load_json(path)
        lines = data.get("lines", [])
        for idx in range(len(lines) - 1):
            first = lines[idx]
            second = lines[idx + 1]
            first_tokens = _normalize_tokens(first.get("text", ""))
            second_tokens = _normalize_tokens(second.get("text", ""))
            if not _is_alternating_pair(first_tokens, second_tokens):
                continue
            rows.append(
                {
                    "gold_path": str(path),
                    "line_index": idx + 1,
                    "first_text": first.get("text", ""),
                    "second_text": second.get("text", ""),
                    "tokens": first_tokens,
                }
            )

    return {
        "gold_root": str(gold_root),
        "match_count": len(rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold-root",
        type=Path,
        default=Path("benchmarks/clip_gold_candidate/20260312T_curated_clips"),
    )
    args = parser.parse_args()
    print(json.dumps(analyze(gold_root=args.gold_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
