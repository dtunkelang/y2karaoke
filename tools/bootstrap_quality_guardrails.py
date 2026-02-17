#!/usr/bin/env python3
"""Quality guardrails for visual-bootstrap gold timing outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_GLOBS = [
    "benchmarks/**/*.visual.gold.json",
    "benchmarks/**/*.karaoke-seed.gold.json",
]


def _iter_candidates(root: Path, patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        out.extend(root.glob(pattern))
    return sorted(set(p for p in out if p.is_file()))


def _is_visual_bootstrap_doc(doc: dict[str, Any]) -> bool:
    return bool(doc.get("visual_suitability")) or bool(doc.get("candidate_url"))


def _extract_confidences(doc: dict[str, Any]) -> tuple[list[float], list[float]]:
    line_conf: list[float] = []
    word_conf: list[float] = []

    for line in doc.get("lines", []):
        lc = line.get("confidence")
        if isinstance(lc, (int, float)):
            line_conf.append(float(lc))
        for word in line.get("words", []):
            wc = word.get("confidence")
            if isinstance(wc, (int, float)):
                word_conf.append(float(wc))
    return line_conf, word_conf


def _check_doc(
    doc: dict[str, Any],
    *,
    min_detectability: float,
    min_word_level_score: float,
    min_word_conf_mean: float,
    min_line_conf_mean: float,
) -> list[str]:
    issues: list[str] = []

    visual = doc.get("visual_suitability", {})
    if visual:
        detectability = float(visual.get("detectability_score", 0.0))
        word_level = float(visual.get("word_level_score", 0.0))
        if detectability < min_detectability:
            issues.append(
                f"detectability_score too low ({detectability:.3f} < {min_detectability:.3f})"
            )
        if word_level < min_word_level_score:
            issues.append(
                f"word_level_score too low ({word_level:.3f} < {min_word_level_score:.3f})"
            )

    line_conf, word_conf = _extract_confidences(doc)
    if not line_conf:
        issues.append("missing line confidence values")
    if not word_conf:
        issues.append("missing word confidence values")

    if line_conf:
        if any(c < 0.0 or c > 1.0 for c in line_conf):
            issues.append("line confidence out of range [0,1]")
        mean_line = sum(line_conf) / len(line_conf)
        if mean_line < min_line_conf_mean:
            issues.append(
                f"mean line confidence too low ({mean_line:.3f} < {min_line_conf_mean:.3f})"
            )

    if word_conf:
        if any(c < 0.0 or c > 1.0 for c in word_conf):
            issues.append("word confidence out of range [0,1]")
        mean_word = sum(word_conf) / len(word_conf)
        if mean_word < min_word_conf_mean:
            issues.append(
                f"mean word confidence too low ({mean_word:.3f} < {min_word_conf_mean:.3f})"
            )

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap quality guardrails")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root (default: current directory)",
    )
    parser.add_argument(
        "--glob",
        action="append",
        default=None,
        help="Glob pattern(s) relative to root. Can be repeated.",
    )
    parser.add_argument("--min-detectability", type=float, default=0.30)
    parser.add_argument("--min-word-level-score", type=float, default=0.10)
    parser.add_argument("--min-word-confidence-mean", type=float, default=0.25)
    parser.add_argument("--min-line-confidence-mean", type=float, default=0.25)
    args = parser.parse_args()

    root = args.root.resolve()
    patterns = args.glob or DEFAULT_GLOBS
    candidates = _iter_candidates(root, patterns)

    analyzed = 0
    failures: list[str] = []

    for path in candidates:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"{path}: invalid JSON ({exc})")
            continue

        if not isinstance(doc, dict) or not _is_visual_bootstrap_doc(doc):
            continue

        analyzed += 1
        issues = _check_doc(
            doc,
            min_detectability=args.min_detectability,
            min_word_level_score=args.min_word_level_score,
            min_word_conf_mean=args.min_word_confidence_mean,
            min_line_conf_mean=args.min_line_confidence_mean,
        )
        if issues:
            failures.append(f"{path}: " + "; ".join(issues))

    if failures:
        print("bootstrap_quality_guardrails: FAIL")
        for line in failures:
            print(f"- {line}")
        return 1

    print(
        f"bootstrap_quality_guardrails: OK (analyzed {analyzed} visual bootstrap file(s))"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
