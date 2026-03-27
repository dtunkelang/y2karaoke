"""Analyze noisy line-local support using text and phonetic similarity."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from y2karaoke.core import phonetic_utils

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _normalize_token(text: str) -> str:
    return "".join(_TOKEN_RE.findall(text.lower()))


def _line_tokens(text: str) -> list[str]:
    return [
        _normalize_token(token) for token in text.split() if _normalize_token(token)
    ]


def _token_scores(
    line_tokens: list[str],
    whisper_tokens: list[str],
    *,
    language: str,
) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for line_token in line_tokens:
        best_match = ""
        best_text = 0.0
        best_phon = 0.0
        best_joint = 0.0
        for whisper_token in whisper_tokens:
            text_score = phonetic_utils._text_similarity_basic(
                line_token, whisper_token
            )
            phon_score = phonetic_utils._phonetic_similarity(
                line_token,
                whisper_token,
                language,
            )
            joint = max(text_score, phon_score)
            if joint > best_joint:
                best_match = whisper_token
                best_text = text_score
                best_phon = phon_score
                best_joint = joint
        scores.append(
            {
                "line_token": line_token,
                "best_match": best_match,
                "best_text_similarity": round(best_text, 3),
                "best_phonetic_similarity": round(best_phon, 3),
                "best_joint_similarity": round(best_joint, 3),
            }
        )
    return scores


def _analyze_line(line: dict[str, Any], *, language: str) -> dict[str, Any]:
    whisper_words = line.get("whisper_window_words", [])
    whisper_tokens = [
        _normalize_token(str(word.get("text") or ""))
        for word in whisper_words
        if _normalize_token(str(word.get("text") or ""))
    ]
    line_tokens = _line_tokens(str(line.get("text") or ""))
    token_scores = _token_scores(line_tokens, whisper_tokens, language=language)
    joint_scores = [float(score["best_joint_similarity"]) for score in token_scores]
    return {
        "index": int(line["index"]),
        "text": str(line["text"]),
        "pred_start": float(line["start"]),
        "pred_end": float(line["end"]),
        "whisper_window_word_count": int(line.get("whisper_window_word_count", 0)),
        "whisper_window_text": " ".join(
            str(word.get("text") or "") for word in whisper_words
        ).strip(),
        "token_scores": token_scores,
        "joint_similarity_mean": round(
            sum(joint_scores) / max(1, len(joint_scores)),
            3,
        ),
        "joint_similarity_min": round(min(joint_scores), 3) if joint_scores else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timing_report", help="Timing report JSON path")
    parser.add_argument(
        "--line",
        type=int,
        action="append",
        dest="line_indexes",
        help="Specific 1-based line index to analyze; repeatable",
    )
    parser.add_argument("--language", default="es", help="Phonetic language code")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    report = _load_json(Path(args.timing_report))
    selected = []
    wanted = set(args.line_indexes or [])
    for line in report.get("lines", []):
        if wanted and int(line["index"]) not in wanted:
            continue
        selected.append(_analyze_line(line, language=args.language))

    payload = {
        "timing_report": str(Path(args.timing_report).resolve()),
        "language": args.language,
        "lines": selected,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"# {payload['timing_report']}")
    for line in selected:
        print(
            f"## line {line['index']} {line['text']} "
            f"(joint_mean={line['joint_similarity_mean']:.3f}, "
            f"joint_min={line['joint_similarity_min']:.3f})"
        )
        print(
            f"- whisper window ({line['whisper_window_word_count']}): "
            f"{line['whisper_window_text']}"
        )
        for score in line["token_scores"]:
            print(
                f"- {score['line_token']} -> {score['best_match']} "
                f"(text={score['best_text_similarity']:.3f}, "
                f"phon={score['best_phonetic_similarity']:.3f}, "
                f"joint={score['best_joint_similarity']:.3f})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
