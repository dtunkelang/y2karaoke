"""Simulate line timing nudges from fuzzy phonetic span support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import analyze_phonetic_line_support as support_tool


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _simulate_line(
    line: dict[str, Any],
    *,
    language: str,
    min_joint_score: float,
) -> dict[str, Any]:
    analysis = support_tool._analyze_line(line, language=language)
    best_span = analysis["best_span"]
    result: dict[str, Any] = {
        "index": analysis["index"],
        "text": analysis["text"],
        "pred_start": analysis["pred_start"],
        "pred_end": analysis["pred_end"],
        "joint_similarity_mean": analysis["joint_similarity_mean"],
        "best_span": best_span,
        "candidate_start": None,
        "candidate_end": None,
        "start_shift": 0.0,
        "end_shift": 0.0,
        "eligible": False,
    }
    span_start = best_span.get("span_start")
    span_end = best_span.get("span_end")
    whisper_words = list(line.get("whisper_window_words") or [])
    if (
        span_start is None
        or span_end is None
        or best_span["joint_score"] < min_joint_score
        or span_start >= len(whisper_words)
        or span_end <= span_start
        or span_end > len(whisper_words)
    ):
        return result
    start_word = whisper_words[span_start]
    end_word = whisper_words[span_end - 1]
    if "start" not in start_word or "end" not in end_word:
        return result
    candidate_start = float(start_word["start"])
    candidate_end = float(end_word["end"])
    result.update(
        {
            "candidate_start": candidate_start,
            "candidate_end": candidate_end,
            "start_shift": round(candidate_start - analysis["pred_start"], 3),
            "end_shift": round(candidate_end - analysis["pred_end"], 3),
            "eligible": True,
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timing_report", help="Timing report JSON path")
    parser.add_argument(
        "--line",
        type=int,
        action="append",
        dest="line_indexes",
        help="Specific 1-based line index to simulate; repeatable",
    )
    parser.add_argument("--language", default="es", help="Phonetic language code")
    parser.add_argument(
        "--min-joint-score",
        type=float,
        default=0.6,
        help="Minimum best-span joint score required for eligibility",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    report = _load_json(Path(args.timing_report))
    wanted = set(args.line_indexes or [])
    lines = [
        _simulate_line(
            line,
            language=args.language,
            min_joint_score=args.min_joint_score,
        )
        for line in report.get("lines", [])
        if not wanted or int(line["index"]) in wanted
    ]
    payload = {
        "timing_report": str(Path(args.timing_report).resolve()),
        "language": args.language,
        "min_joint_score": args.min_joint_score,
        "lines": lines,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"# {payload['timing_report']}")
    for line in lines:
        print(
            f"## line {line['index']} {line['text']} "
            f"(eligible={line['eligible']}, span_joint={line['best_span']['joint_score']:.3f})"
        )
        print(f"- predicted: {line['pred_start']:.2f}-{line['pred_end']:.2f}")
        print(
            f"- best span: {line['best_span']['span_text']} "
            f"(text={line['best_span']['text_similarity']:.3f}, "
            f"phon={line['best_span']['phonetic_similarity_mean']:.3f}, "
            f"joint={line['best_span']['joint_score']:.3f})"
        )
        if not line["eligible"]:
            continue
        print(
            f"- candidate: {line['candidate_start']:.2f}-{line['candidate_end']:.2f} "
            f"(shift {line['start_shift']:+.2f}/{line['end_shift']:+.2f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
