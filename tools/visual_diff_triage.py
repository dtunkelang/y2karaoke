#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import evaluate_visual_lyrics_quality as ev  # type: ignore
from y2karaoke.core.components.lyrics.lrc import parse_lrc_with_timing
from y2karaoke.core.text_utils import normalize_text_basic


@dataclass
class TokenRec:
    token: str
    source: str  # ref|ext
    token_index: int
    line_index: int
    line_text: str
    line_key: str
    start: float | None
    end: float | None


def _load_eval_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_lrc_path(payload: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override
    ref = payload.get("reference_source", {}) or {}
    if ref.get("type") == "lrc_file" and ref.get("path"):
        return Path(str(ref["path"]))
    raise SystemExit(
        "ERROR: eval payload does not contain local lrc_file path; pass --lrc-file"
    )


def _load_extracted_lines_with_words(gold_path: Path) -> list[dict[str, Any]]:
    doc = json.loads(gold_path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for idx, line in enumerate(doc.get("lines", []), start=1):
        words = line.get("words", []) or []
        token_recs: list[dict[str, Any]] = []
        flat_tokens: list[str] = []
        for w in words:
            raw = str(w.get("text", "") or "")
            norm = normalize_text_basic(raw)
            if not norm:
                continue
            for tok in norm.split():
                flat_tokens.append(tok)
                token_recs.append(
                    {
                        "token": tok,
                        "start": (
                            float(w.get("start"))
                            if w.get("start") is not None
                            else None
                        ),
                        "end": (
                            float(w.get("end")) if w.get("end") is not None else None
                        ),
                    }
                )
        if not flat_tokens:
            continue
        out.append(
            {
                "line_index": idx,
                "line_text": str(line.get("text", "") or ""),
                "line_key": normalize_text_basic(" ".join(flat_tokens)).strip(),
                "tokens": token_recs,
                "start": (
                    float(line.get("start")) if line.get("start") is not None else None
                ),
                "end": float(line.get("end")) if line.get("end") is not None else None,
            }
        )
    return out


def _load_reference_lines_with_times(
    *, lrc_path: Path, title: str, artist: str, include_parenthetical: bool
) -> list[dict[str, Any]]:
    lrc_text = lrc_path.read_text(encoding="utf-8")
    timed = parse_lrc_with_timing(lrc_text, title=title, artist=artist)
    out: list[dict[str, Any]] = []
    for idx, (start, text) in enumerate(timed, start=1):
        token_parts = ev._split_tokens_with_optional_flags(text)
        toks: list[str] = []
        for rec in token_parts:
            if not include_parenthetical and bool(rec["optional"]):
                continue
            toks.append(str(rec["token"]))
        if not toks:
            continue
        line_key = normalize_text_basic(" ".join(toks)).strip()
        if not line_key:
            continue
        norm_toks: list[str] = []
        for tok in toks:
            norm = normalize_text_basic(tok)
            if not norm:
                continue
            norm_toks.extend(norm.split())
        if not norm_toks:
            continue
        out.append(
            {
                "line_index": idx,
                "line_text": str(text),
                "line_key": line_key,
                "tokens": [
                    {"token": t, "start": float(start), "end": None} for t in norm_toks
                ],
                "start": float(start),
                "end": None,
            }
        )
    return out


def _flatten_token_records(lines: list[dict[str, Any]], source: str) -> list[TokenRec]:
    out: list[TokenRec] = []
    token_idx = 0
    for line in lines:
        for t in line.get("tokens", []):
            out.append(
                TokenRec(
                    token=str(t["token"]),
                    source=source,
                    token_index=token_idx,
                    line_index=int(line["line_index"]),
                    line_text=str(line["line_text"]),
                    line_key=str(line["line_key"]),
                    start=(None if t.get("start") is None else float(t["start"])),
                    end=(None if t.get("end") is None else float(t["end"])),
                )
            )
            token_idx += 1
    return out


def _build_repeat_capped_token_records(
    reference_lines: list[dict[str, Any]], cap_line_keys: list[str], source: str
) -> list[TokenRec]:
    cap_counts: dict[str, int] = {}
    for key in cap_line_keys:
        cap_counts[key] = cap_counts.get(key, 0) + 1
    seen: dict[str, int] = {}
    kept_lines: list[dict[str, Any]] = []
    for line in reference_lines:
        key = str(line.get("line_key", ""))
        if not key:
            continue
        seen[key] = seen.get(key, 0) + 1
        allowed = max(1, cap_counts.get(key, 0))
        if seen[key] > allowed:
            continue
        kept_lines.append(line)
    return _flatten_token_records(kept_lines, source)


def _line_context(
    lines: list[dict[str, Any]], line_idxs: set[int], window: int = 1
) -> list[dict[str, Any]]:
    if not lines or not line_idxs:
        return []
    idx_map = {int(line["line_index"]): pos for pos, line in enumerate(lines)}
    positions: set[int] = set()
    for li in line_idxs:
        pos = idx_map.get(li)
        if pos is None:
            continue
        for p in range(max(0, pos - window), min(len(lines), pos + window + 1)):
            positions.add(p)
    out: list[dict[str, Any]] = []
    for p in sorted(positions):
        line = lines[p]
        out.append(
            {
                "line_index": int(line["line_index"]),
                "start": line.get("start"),
                "end": line.get("end"),
                "text": str(line.get("line_text", "")),
                "line_key": str(line.get("line_key", "")),
            }
        )
    return out


def _infer_tags(diff: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    ref_preview = str(diff.get("ref_preview", "")).lower()
    ext_preview = str(diff.get("ext_preview", "")).lower()
    ext_count = int(diff.get("ext_count", 0) or 0)
    ref_count = int(diff.get("ref_count", 0) or 0)
    if any(
        tok in (ref_preview + " " + ext_preview)
        for tok in ["doh", "dohi", "uh", "come on"]
    ):
        tags.append("chant_or_adlib")
    if any(tok in (ref_preview + " " + ext_preview) for tok in ["l've", "1'v", "i've"]):
        tags.append("ocr_contraction_confusion")
    if ext_count == 0 and ref_count >= 10:
        tags.append("missing_block")
    if ref_count == 0 and ext_count >= 10:
        tags.append("extra_block")
    if diff.get("op") == "replace" and ref_count >= 20 and ext_count <= 2:
        tags.append("catastrophic_alignment_collapse")
    if any(
        tok in (ref_preview + " " + ext_preview)
        for tok in ["uptown funk", "counting stars"]
    ):
        tags.append("chorus_repeat_region")
    return tags


def _summarize_opcodes(
    reference_tokens: list[str], extracted_tokens: list[str]
) -> list[tuple[str, int, int, int, int]]:
    matcher = SequenceMatcher(a=reference_tokens, b=extracted_tokens, autojunk=False)
    return matcher.get_opcodes()


def _find_diff_opcode(
    opcodes: list[tuple[str, int, int, int, int]], diff: dict[str, Any]
) -> tuple[str, int, int, int, int] | None:
    target = (
        str(diff.get("op")),
        int(diff.get("ref_range", [0, 0])[0]),
        int(diff.get("ref_range", [0, 0])[1]),
        int(diff.get("ext_range", [0, 0])[0]),
        int(diff.get("ext_range", [0, 0])[1]),
    )
    for op in opcodes:
        if op == target:
            return op
    return None


def _token_span_meta(tokens: list[TokenRec], start: int, end: int) -> dict[str, Any]:
    if start >= end or start < 0 or end > len(tokens):
        return {
            "token_range": [start, end],
            "line_indexes": [],
            "time_span": [None, None],
        }
    span = tokens[start:end]
    line_idxs = sorted({t.line_index for t in span})
    starts = [t.start for t in span if t.start is not None]
    ends = [t.end for t in span if t.end is not None]
    if starts:
        t0 = min(starts)
    else:
        t0 = None
    if ends:
        t1 = max(ends)
    elif starts:
        t1 = max(starts)
    else:
        t1 = None
    return {
        "token_range": [start, end],
        "line_indexes": line_idxs,
        "time_span": [t0, t1],
    }


def _build_metric_triage(
    *,
    metric_name: str,
    diffs: list[dict[str, Any]],
    opcodes: list[tuple[str, int, int, int, int]],
    ref_tokens: list[TokenRec],
    ext_tokens: list[TokenRec],
    ref_lines: list[dict[str, Any]],
    ext_lines: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rank, diff in enumerate(diffs[:limit], start=1):
        op = _find_diff_opcode(opcodes, diff)
        if op is None:
            ref_i1, ref_i2 = [int(x) for x in diff["ref_range"]]
            ext_j1, ext_j2 = [int(x) for x in diff["ext_range"]]
        else:
            _op, ref_i1, ref_i2, ext_j1, ext_j2 = op
        ref_span = _token_span_meta(ref_tokens, ref_i1, ref_i2)
        ext_span = _token_span_meta(ext_tokens, ext_j1, ext_j2)
        ref_ctx = _line_context(ref_lines, set(ref_span["line_indexes"]), window=1)
        ext_ctx = _line_context(ext_lines, set(ext_span["line_indexes"]), window=1)
        out.append(
            {
                "rank": rank,
                "metric": metric_name,
                "magnitude": int(diff.get("magnitude", 0) or 0),
                "op": str(diff.get("op")),
                "tags": _infer_tags(diff),
                "ref": {
                    **ref_span,
                    "preview": str(diff.get("ref_preview", "")),
                    "count": int(diff.get("ref_count", 0) or 0),
                    "line_context": ref_ctx,
                },
                "ext": {
                    **ext_span,
                    "preview": str(diff.get("ext_preview", "")),
                    "count": int(diff.get("ext_count", 0) or 0),
                    "line_context": ext_ctx,
                },
                "triage": {
                    "label": None,
                    "notes": "",
                    "likely_reference_mismatch": None,
                },
            }
        )
    return out


def _fmt_time(t: Any) -> str:
    if t is None:
        return "-"
    try:
        return f"{float(t):.2f}s"
    except Exception:
        return "-"


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append(f"# Visual Diff Triage: {payload['artist']} - {payload['title']}")
    lines.append("")
    lines.append(f"Gold: `{payload['gold_json']}`")
    lines.append(f"LRC: `{payload['lrc_file']}`")
    lines.append("")
    for metric in ("strict", "repeat_capped"):
        m = payload["metrics"][metric]
        lines.append(f"## {metric}")
        lines.append(
            f"f1={m['f1']:.4f} precision={m['precision']:.4f} recall={m['recall']:.4f} matched={m['matched_token_count']}/{m['reference_token_count']} ext={m['extracted_token_count']}"
        )
        lines.append("")
        for block in payload["triage_blocks"][metric]:
            lines.append(
                f"### #{block['rank']} {block['op']} magnitude={block['magnitude']} tags={', '.join(block['tags']) or 'none'}"
            )
            lines.append(
                f"Ref tokens {block['ref']['token_range'][0]}:{block['ref']['token_range'][1]} ({block['ref']['count']}) time {_fmt_time(block['ref']['time_span'][0])}-{_fmt_time(block['ref']['time_span'][1])}"
            )
            lines.append(f"Ref preview: {block['ref']['preview']}")
            lines.append(
                f"Ext tokens {block['ext']['token_range'][0]}:{block['ext']['token_range'][1]} ({block['ext']['count']}) time {_fmt_time(block['ext']['time_span'][0])}-{_fmt_time(block['ext']['time_span'][1])}"
            )
            lines.append(f"Ext preview: {block['ext']['preview']}")
            lines.append("")
            lines.append("Ref context:")
            for ctx in block["ref"]["line_context"]:
                lines.append(
                    f"- [{ctx['line_index']}] {_fmt_time(ctx['start'])}-{_fmt_time(ctx['end'])} {ctx['text']}"
                )
            lines.append("Ext context:")
            for ctx in block["ext"]["line_context"]:
                lines.append(
                    f"- [{ctx['line_index']}] {_fmt_time(ctx['start'])}-{_fmt_time(ctx['end'])} {ctx['text']}"
                )
            lines.append("Triage:")
            lines.append("- Label:")
            lines.append("- Likely reference mismatch?:")
            lines.append("- Notes:")
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate manual triage report for visual eval diff blocks."
    )
    p.add_argument("--eval-json", type=Path, required=True)
    p.add_argument("--gold-json", type=Path, default=None)
    p.add_argument("--lrc-file", type=Path, default=None)
    p.add_argument("--max-diff-blocks", type=int, default=8)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-md", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    eval_payload = _load_eval_payload(args.eval_json)
    gold_path = args.gold_json or Path(str(eval_payload["gold_json"]))
    lrc_path = _resolve_lrc_path(eval_payload, args.lrc_file)
    title = str(eval_payload["title"])
    artist = str(eval_payload["artist"])
    include_parenthetical = (
        str(eval_payload.get("lrc_mode", "optional")) == "include_parenthetical"
    )

    ext_lines = _load_extracted_lines_with_words(gold_path)
    ref_lines = _load_reference_lines_with_times(
        lrc_path=lrc_path,
        title=title,
        artist=artist,
        include_parenthetical=include_parenthetical,
    )

    ext_tokens = _flatten_token_records(ext_lines, "ext")
    ref_tokens = _flatten_token_records(ref_lines, "ref")
    ext_token_texts = [t.token for t in ext_tokens]
    ref_token_texts = [t.token for t in ref_tokens]

    ext_line_keys = [
        str(line.get("line_key", "")) for line in ext_lines if line.get("line_key")
    ]
    ref_line_keys = [
        str(line.get("line_key", "")) for line in ref_lines if line.get("line_key")
    ]
    repeat_ref_tokens = _build_repeat_capped_token_records(
        ref_lines, ext_line_keys, "ref"
    )
    repeat_ext_tokens = _build_repeat_capped_token_records(
        ext_lines, ref_line_keys, "ext"
    )

    strict_opcodes = _summarize_opcodes(ref_token_texts, ext_token_texts)
    repeat_opcodes = _summarize_opcodes(
        [t.token for t in repeat_ref_tokens], [t.token for t in repeat_ext_tokens]
    )

    out = {
        "title": title,
        "artist": artist,
        "eval_json": str(args.eval_json),
        "gold_json": str(gold_path),
        "lrc_file": str(lrc_path),
        "metrics": {
            "strict": eval_payload["strict"],
            "repeat_capped": eval_payload["repeat_capped"],
        },
        "triage_blocks": {
            "strict": _build_metric_triage(
                metric_name="strict",
                diffs=list(eval_payload["strict"].get("largest_diffs", [])),
                opcodes=strict_opcodes,
                ref_tokens=ref_tokens,
                ext_tokens=ext_tokens,
                ref_lines=ref_lines,
                ext_lines=ext_lines,
                limit=max(1, int(args.max_diff_blocks)),
            ),
            "repeat_capped": _build_metric_triage(
                metric_name="repeat_capped",
                diffs=list(eval_payload["repeat_capped"].get("largest_diffs", [])),
                opcodes=repeat_opcodes,
                ref_tokens=repeat_ref_tokens,
                ext_tokens=repeat_ext_tokens,
                ref_lines=ref_lines,
                ext_lines=ext_lines,
                limit=max(1, int(args.max_diff_blocks)),
            ),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    _write_markdown(args.output_md, out)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
