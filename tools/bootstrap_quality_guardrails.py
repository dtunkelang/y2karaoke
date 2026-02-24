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


def _iter_visual_eval_rows(summary_doc: dict[str, Any]) -> list[dict[str, Any]]:
    songs = summary_doc.get("songs", [])
    if not isinstance(songs, list):
        return []
    return [row for row in songs if isinstance(row, dict)]


def _check_visual_eval_summary(
    summary_doc: dict[str, Any],
    *,
    min_strict_f1: float | None,
    min_repeat_capped_f1: float | None,
    min_strict_f1_mean: float | None = None,
    min_repeat_capped_f1_mean: float | None = None,
    min_strict_f1_median: float | None = None,
    min_repeat_capped_f1_median: float | None = None,
) -> list[str]:
    issues: list[str] = []
    if (
        min_strict_f1 is None
        and min_repeat_capped_f1 is None
        and min_strict_f1_mean is None
        and min_repeat_capped_f1_mean is None
        and min_strict_f1_median is None
        and min_repeat_capped_f1_median is None
    ):
        return issues

    summary_block = summary_doc.get("summary", {})
    if not isinstance(summary_block, dict):
        summary_block = {}
    summary_pairs = [
        ("strict_f1_mean", min_strict_f1_mean, "strict_f1_mean"),
        (
            "repeat_capped_f1_mean",
            min_repeat_capped_f1_mean,
            "repeat_capped_f1_mean",
        ),
        ("strict_f1_median", min_strict_f1_median, "strict_f1_median"),
        (
            "repeat_capped_f1_median",
            min_repeat_capped_f1_median,
            "repeat_capped_f1_median",
        ),
    ]
    for label, threshold, key in summary_pairs:
        if threshold is None:
            continue
        val = summary_block.get(key)
        if not isinstance(val, (int, float)) or float(val) < threshold:
            issues.append(f"visual eval {label} too low ({val!r} < {threshold:.3f})")

    for row in _iter_visual_eval_rows(summary_doc):
        if row.get("status") != "ok":
            continue
        label = f"{row.get('index', '?'):02d} {row.get('artist', '?')} - {row.get('title', '?')}"
        if min_strict_f1 is not None:
            strict = row.get("strict", {})
            val = strict.get("f1") if isinstance(strict, dict) else None
            if not isinstance(val, (int, float)) or float(val) < min_strict_f1:
                issues.append(
                    f"visual eval strict f1 too low for {label} ({val!r} < {min_strict_f1:.3f})"
                )
        if min_repeat_capped_f1 is not None:
            rc = row.get("repeat_capped", {})
            val = rc.get("f1") if isinstance(rc, dict) else None
            if not isinstance(val, (int, float)) or float(val) < min_repeat_capped_f1:
                issues.append(
                    "visual eval repeat_capped f1 too low for "
                    f"{label} ({val!r} < {min_repeat_capped_f1:.3f})"
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
    parser.add_argument(
        "--visual-eval-summary-json",
        type=Path,
        default=None,
        help="Optional aggregate visual eval summary JSON from run_visual_eval.py",
    )
    parser.add_argument(
        "--min-visual-eval-strict-f1",
        type=float,
        default=None,
        help="Optional per-song minimum strict F1 gate using --visual-eval-summary-json",
    )
    parser.add_argument(
        "--min-visual-eval-repeat-capped-f1",
        type=float,
        default=None,
        help="Optional per-song minimum repeat-capped F1 gate using --visual-eval-summary-json",
    )
    parser.add_argument("--min-visual-eval-strict-f1-mean", type=float, default=None)
    parser.add_argument(
        "--min-visual-eval-repeat-capped-f1-mean", type=float, default=None
    )
    parser.add_argument("--min-visual-eval-strict-f1-median", type=float, default=None)
    parser.add_argument(
        "--min-visual-eval-repeat-capped-f1-median", type=float, default=None
    )
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

    if args.visual_eval_summary_json is not None:
        try:
            summary_doc = json.loads(
                args.visual_eval_summary_json.read_text(encoding="utf-8")
            )
        except Exception as exc:
            print("bootstrap_quality_guardrails: FAIL")
            print(f"- invalid visual eval summary JSON ({exc})")
            return 1

        eval_issues = _check_visual_eval_summary(
            summary_doc if isinstance(summary_doc, dict) else {},
            min_strict_f1=args.min_visual_eval_strict_f1,
            min_repeat_capped_f1=args.min_visual_eval_repeat_capped_f1,
            min_strict_f1_mean=args.min_visual_eval_strict_f1_mean,
            min_repeat_capped_f1_mean=args.min_visual_eval_repeat_capped_f1_mean,
            min_strict_f1_median=args.min_visual_eval_strict_f1_median,
            min_repeat_capped_f1_median=args.min_visual_eval_repeat_capped_f1_median,
        )
        if eval_issues:
            print("bootstrap_quality_guardrails: FAIL")
            for line in eval_issues:
                print(f"- {line}")
            return 1

    print(
        f"bootstrap_quality_guardrails: OK (analyzed {analyzed} visual bootstrap file(s))"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
