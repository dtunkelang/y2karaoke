#!/usr/bin/env python3
"""Scan cached clip lyrics for adjacent alternating 3-word hook pairs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


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


def analyze(*, root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    pair_counts: Counter[tuple[str, str]] = Counter()
    stem_counts: Counter[str] = Counter()

    for path in sorted(root.rglob("*clip_lyrics.txt")):
        try:
            lines = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except OSError:
            continue
        stem = path.name.removesuffix("_clip_lyrics.txt")
        for idx in range(len(lines) - 1):
            first = lines[idx]
            second = lines[idx + 1]
            if not _is_alternating_pair(
                _normalize_tokens(first),
                _normalize_tokens(second),
            ):
                continue
            rows.append(
                {
                    "path": str(path),
                    "line_index": idx + 1,
                    "first_text": first,
                    "second_text": second,
                    "clip_stem": stem,
                }
            )
            pair_counts[(first, second)] += 1
            stem_counts[stem] += 1

    return {
        "root": str(root),
        "hit_count": len(rows),
        "distinct_pairs": [
            {"first_text": first, "second_text": second, "count": count}
            for (first, second), count in pair_counts.most_common()
        ],
        "distinct_clip_stems": [
            {"clip_stem": stem, "count": count}
            for stem, count in stem_counts.most_common()
        ],
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()
    print(json.dumps(analyze(root=args.root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
