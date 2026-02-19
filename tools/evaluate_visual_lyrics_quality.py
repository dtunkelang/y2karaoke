#!/usr/bin/env python3
"""Evaluate visual-bootstrapped lyrics token quality against LRC text."""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from y2karaoke.core.components.lyrics.lrc import parse_lrc_with_timing  # noqa: E402
from y2karaoke.core.components.lyrics.sync import fetch_lyrics_multi_source  # noqa: E402
from y2karaoke.core.text_utils import normalize_text_basic  # noqa: E402


def _split_tokens_with_optional_flags(text: str) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    if not text:
        return tokens

    depth = 0
    cur: list[str] = []
    token_optional = False
    for ch in text:
        if ch == "(":
            if cur:
                token = "".join(cur)
                tokens.append({"token": token, "optional": token_optional})
                cur = []
            depth += 1
            continue
        if ch == ")":
            if cur:
                token = "".join(cur)
                tokens.append({"token": token, "optional": token_optional})
                cur = []
            depth = max(0, depth - 1)
            continue
        if ch.isalnum() or ch == "'":
            if not cur:
                token_optional = depth > 0
            cur.append(ch)
            continue
        if cur:
            token = "".join(cur)
            tokens.append({"token": token, "optional": token_optional})
            cur = []

    if cur:
        token = "".join(cur)
        tokens.append({"token": token, "optional": token_optional})

    out: list[dict[str, Any]] = []
    for rec in tokens:
        norm = normalize_text_basic(str(rec["token"]))
        if not norm:
            continue
        for tok in norm.split():
            out.append({"token": tok, "optional": bool(rec["optional"])})
    return out


def _load_extracted_tokens(gold_path: Path) -> list[str]:
    doc = json.loads(gold_path.read_text(encoding="utf-8"))
    out: list[str] = []
    for line in doc.get("lines", []):
        for word in line.get("words", []):
            token = normalize_text_basic(str(word.get("text", "")))
            if not token:
                continue
            out.extend(token.split())
    return out


def _load_lrc_tokens(
    *,
    title: str,
    artist: str,
    include_parenthetical: bool,
) -> list[str]:
    lrc_text, _, _ = fetch_lyrics_multi_source(title, artist, synced_only=True)
    if not lrc_text:
        return []

    timed_lines = parse_lrc_with_timing(lrc_text, title=title, artist=artist)
    out: list[str] = []
    for _, text in timed_lines:
        token_recs = _split_tokens_with_optional_flags(text)
        for rec in token_recs:
            if not include_parenthetical and bool(rec["optional"]):
                continue
            out.append(str(rec["token"]))
    return out


def _summarize_alignment(
    reference_tokens: list[str],
    extracted_tokens: list[str],
    *,
    max_diff_blocks: int,
) -> dict[str, Any]:
    matcher = SequenceMatcher(a=reference_tokens, b=extracted_tokens, autojunk=False)
    opcodes = matcher.get_opcodes()
    matched = sum(i2 - i1 for op, i1, i2, _j1, _j2 in opcodes if op == "equal")
    precision = (matched / len(extracted_tokens)) if extracted_tokens else 0.0
    recall = (matched / len(reference_tokens)) if reference_tokens else 0.0
    f1 = (
        (2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0.0
        else 0.0
    )

    diffs: list[dict[str, Any]] = []
    for op, i1, i2, j1, j2 in opcodes:
        if op == "equal":
            continue
        ref_slice = reference_tokens[i1:i2]
        ext_slice = extracted_tokens[j1:j2]
        diffs.append(
            {
                "op": op,
                "ref_range": [i1, i2],
                "ext_range": [j1, j2],
                "ref_count": len(ref_slice),
                "ext_count": len(ext_slice),
                "ref_preview": " ".join(ref_slice[:20]),
                "ext_preview": " ".join(ext_slice[:20]),
                "magnitude": max(len(ref_slice), len(ext_slice)),
            }
        )
    diffs.sort(key=lambda rec: int(rec["magnitude"]), reverse=True)

    return {
        "reference_token_count": len(reference_tokens),
        "extracted_token_count": len(extracted_tokens),
        "matched_token_count": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "largest_diffs": diffs[:max_diff_blocks],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare extracted visual lyrics to LRC token order."
    )
    parser.add_argument("--gold-json", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--artist", required=True)
    parser.add_argument(
        "--include-parenthetical-lrc",
        action="store_true",
        help="Include parenthetical LRC tokens in reference (default: optional/ignored).",
    )
    parser.add_argument("--max-diff-blocks", type=int, default=8)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.gold_json.exists():
        print(f"ERROR: missing gold file: {args.gold_json}")
        return 2

    extracted_tokens = _load_extracted_tokens(args.gold_json)
    reference_tokens = _load_lrc_tokens(
        title=args.title,
        artist=args.artist,
        include_parenthetical=bool(args.include_parenthetical_lrc),
    )
    if not reference_tokens:
        print(
            f"ERROR: no synced LRC tokens found for '{args.artist}' - '{args.title}'."
        )
        return 3

    summary = _summarize_alignment(
        reference_tokens,
        extracted_tokens,
        max_diff_blocks=max(1, int(args.max_diff_blocks)),
    )
    payload = {
        "title": args.title,
        "artist": args.artist,
        "gold_json": str(args.gold_json),
        "lrc_mode": (
            "include_parenthetical" if args.include_parenthetical_lrc else "optional"
        ),
        **summary,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"precision={summary['precision']:.4f} "
        f"recall={summary['recall']:.4f} "
        f"f1={summary['f1']:.4f} "
        f"matched={summary['matched_token_count']}/"
        f"{summary['reference_token_count']} "
        f"ext={summary['extracted_token_count']}"
    )
    if payload["largest_diffs"]:
        print("largest_diffs:")
        for diff in payload["largest_diffs"]:
            print(
                f"- {diff['op']} ref[{diff['ref_range'][0]}:{diff['ref_range'][1]}] "
                f"ext[{diff['ext_range'][0]}:{diff['ext_range'][1]}] "
                f"(ref={diff['ref_count']}, ext={diff['ext_count']})"
            )
            print(f"  ref: {diff['ref_preview']}")
            print(f"  ext: {diff['ext_preview']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
