#!/usr/bin/env python3
"""Recommend prioritized human-guided correction tasks from benchmark output."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Any
import unicodedata


def _resolve_report(path: Path) -> Path:
    return path / "benchmark_report.json" if path.is_dir() else path


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(_resolve_report(path).read_text(encoding="utf-8"))


def _song_name(song: dict[str, Any]) -> str:
    return f"{song.get('artist', '')} - {song.get('title', '')}".strip()


def _normalize_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    folded = "".join(
        ch
        for ch in unicodedata.normalize("NFKD", text.lower())
        if unicodedata.category(ch) != "Mn"
    )
    folded = re.sub(r"[’`´]", "'", folded)
    folded = re.sub(r"[^a-z0-9'\s]", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def _text_similarity(left: Any, right: Any) -> float:
    a = _normalize_text(left)
    b = _normalize_text(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _token_overlap(left: Any, right: Any) -> float:
    a = set(_normalize_text(left).split())
    b = set(_normalize_text(right).split())
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


def _collect_mismatch_examples(song: dict[str, Any], *, limit: int = 2) -> list[str]:
    report_path_raw = song.get("report_path")
    if not isinstance(report_path_raw, str) or not report_path_raw:
        return []
    report_path = Path(report_path_raw)
    if not report_path.exists():
        return []
    try:
        report_doc = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    lines = report_doc.get("lines", [])
    if not isinstance(lines, list):
        return []
    examples: list[str] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        line_text = line.get("text")
        anchor_text = line.get("nearest_segment_start_text")
        line_start = line.get("start")
        anchor_start = line.get("nearest_segment_start")
        if not isinstance(line_text, str) or not isinstance(anchor_text, str):
            continue
        if not isinstance(line_start, (int, float)) or not isinstance(
            anchor_start, (int, float)
        ):
            continue
        start_delta = abs(float(line_start) - float(anchor_start))
        sim = _text_similarity(line_text, anchor_text)
        overlap = _token_overlap(line_text, anchor_text)
        # Focus examples on likely lexical mismatches that are still temporally close
        # enough for fast manual correction (snap + nudge).
        if start_delta > 0.3:
            continue
        if sim >= 0.58:
            continue
        if overlap < 0.45:
            continue
        examples.append(
            f"line='{line_text[:80]}' vs anchor='{anchor_text[:80]}' "
            f"(delta={start_delta:.2f}s, sim={sim:.2f}, overlap={overlap:.2f})"
        )
        if len(examples) >= limit:
            break
    return examples


def _collect_correction_targets(song: dict[str, Any], *, limit: int = 2) -> list[str]:
    report_path_raw = song.get("report_path")
    if not isinstance(report_path_raw, str) or not report_path_raw:
        return []
    report_path = Path(report_path_raw)
    if not report_path.exists():
        return []
    try:
        report_doc = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    lines = report_doc.get("lines", [])
    if not isinstance(lines, list):
        return []

    targets: list[str] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        index_raw = line.get("index")
        line_index = int(index_raw) if isinstance(index_raw, (int, float)) else -1
        line_text = str(line.get("text") or "")
        anchor_text = str(line.get("nearest_segment_start_text") or "")
        line_start = line.get("start")
        anchor_start = line.get("nearest_segment_start")
        window_word_count_raw = line.get("whisper_window_word_count")
        window_word_count = (
            int(window_word_count_raw)
            if isinstance(window_word_count_raw, (int, float))
            else 0
        )
        word_slots = line.get("word_slots")
        line_word_count = (
            int(word_slots)
            if isinstance(word_slots, (int, float))
            else len(line_text.split())
        )
        if line_word_count < 1:
            line_word_count = 1

        if not isinstance(line_start, (int, float)) or not isinstance(
            anchor_start, (int, float)
        ):
            continue
        start_delta = abs(float(line_start) - float(anchor_start))
        sim = _text_similarity(line_text, anchor_text)
        overlap = _token_overlap(line_text, anchor_text)

        reason = ""
        if start_delta > 0.8:
            reason = "high_start_delta"
        elif sim < 0.58 and overlap >= 0.45 and start_delta <= 0.3:
            reason = "likely_lexical_mismatch"
        elif line_word_count >= 6 and window_word_count < max(
            2, int(0.35 * line_word_count)
        ):
            reason = "sparse_local_anchor_evidence"
        if not reason:
            continue
        targets.append(
            f"line_index={line_index} reason={reason} delta={start_delta:.2f}s "
            f"sim={sim:.2f} overlap={overlap:.2f} text='{line_text[:70]}'"
        )
        if len(targets) >= limit:
            break
    return targets


def _build_actions(song: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    agreement_cov = float(metrics.get("agreement_coverage_ratio", 0.0) or 0.0)
    agreement_p95 = float(metrics.get("agreement_start_p95_abs_sec", 0.0) or 0.0)
    low_conf = float(metrics.get("low_confidence_ratio", 0.0) or 0.0)
    dtw_line = float(metrics.get("dtw_line_coverage", 0.0) or 0.0)
    fallback_attempted = int(metrics.get("fallback_map_attempted", 0) or 0)
    fallback_selected = int(metrics.get("fallback_map_selected", 0) or 0)

    if agreement_cov < 0.35:
        actions.append(
            "Use gold editor jump-to-anchor + snap-to-onset to label comparable lines faster."
        )
    if agreement_p95 > 0.9:
        actions.append(
            "Micro-nudge line/word timings with hotkeys; focus on first word onset per line."
        )
    if low_conf > 0.08:
        actions.append(
            "Review low-confidence sections first; add word-level corrections around unclear syllables."
        )
    if dtw_line < 0.9:
        actions.append(
            "Inspect timing-source quality (LRC drift/offset) and anchor key phrase starts manually."
        )
    if fallback_attempted > 0 and fallback_selected == 0:
        actions.append(
            "Check fallback-map rejection reason in diagnostics before broad timing edits."
        )
    if not actions:
        actions.append(
            "No urgent manual correction needed; keep as spot-check candidate."
        )
    return actions


def _score_song(metrics: dict[str, Any]) -> float:
    agreement_cov = float(metrics.get("agreement_coverage_ratio", 0.0) or 0.0)
    agreement_p95 = float(metrics.get("agreement_start_p95_abs_sec", 0.0) or 0.0)
    low_conf = float(metrics.get("low_confidence_ratio", 0.0) or 0.0)
    dtw_line = float(metrics.get("dtw_line_coverage", 0.0) or 0.0)
    fallback_attempted = float(metrics.get("fallback_map_attempted", 0.0) or 0.0)
    fallback_selected = float(metrics.get("fallback_map_selected", 0.0) or 0.0)
    fallback_penalty = max(0.0, fallback_attempted - fallback_selected)
    # Higher score = better candidate for human correction priority.
    return (
        (max(0.0, 0.4 - agreement_cov) * 2.5)
        + (max(0.0, agreement_p95 - 0.8) * 1.6)
        + (low_conf * 1.4)
        + (max(0.0, 0.9 - dtw_line) * 1.2)
        + (fallback_penalty * 0.08)
    )


def _recommend(doc: dict[str, Any], top: int, min_priority: float) -> dict[str, Any]:
    songs = doc.get("songs", []) or []
    rows: list[dict[str, Any]] = []
    for song in songs:
        if not isinstance(song, dict):
            continue
        metrics_raw = song.get("metrics", {}) or {}
        metrics = metrics_raw if isinstance(metrics_raw, dict) else {}
        score = _score_song(metrics)
        rows.append(
            {
                "song": _song_name(song),
                "status": str(song.get("status", "")),
                "priority_score": round(score, 3),
                "agreement_coverage_ratio": round(
                    float(metrics.get("agreement_coverage_ratio", 0.0) or 0.0), 3
                ),
                "agreement_start_p95_abs_sec": round(
                    float(metrics.get("agreement_start_p95_abs_sec", 0.0) or 0.0), 3
                ),
                "low_confidence_ratio": round(
                    float(metrics.get("low_confidence_ratio", 0.0) or 0.0), 3
                ),
                "dtw_line_coverage": round(
                    float(metrics.get("dtw_line_coverage", 0.0) or 0.0), 3
                ),
                "actions": _build_actions(song, metrics),
                "mismatch_examples": _collect_mismatch_examples(song),
                "suggested_targets": _collect_correction_targets(song),
            }
        )
    rows = [row for row in rows if row.get("status") == "ok"]
    rows = [
        row
        for row in rows
        if float(row.get("priority_score", 0.0) or 0.0) >= min_priority
    ]
    rows.sort(key=lambda row: float(row.get("priority_score", 0.0)), reverse=True)
    if top > 0:
        rows = rows[:top]
    return {
        "song_count_considered": len(rows),
        "top": top,
        "min_priority": min_priority,
        "rows": rows,
    }


def _validate_payload_schema(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    for key in ("song_count_considered", "top"):
        if not isinstance(payload.get(key), int):
            raise ValueError(f"payload.{key} must be an integer")
    if not isinstance(payload.get("min_priority"), (int, float)):
        raise ValueError("payload.min_priority must be numeric")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("payload.rows must be an array")
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"payload.rows[{idx}] must be an object")
        for key in (
            "song",
            "status",
            "priority_score",
            "agreement_coverage_ratio",
            "agreement_start_p95_abs_sec",
            "low_confidence_ratio",
            "dtw_line_coverage",
            "actions",
            "mismatch_examples",
            "suggested_targets",
        ):
            if key not in row:
                raise ValueError(f"payload.rows[{idx}].{key} is required")
        if not isinstance(row.get("song"), str):
            raise ValueError(f"payload.rows[{idx}].song must be a string")
        if not isinstance(row.get("status"), str):
            raise ValueError(f"payload.rows[{idx}].status must be a string")
        for num_key in (
            "priority_score",
            "agreement_coverage_ratio",
            "agreement_start_p95_abs_sec",
            "low_confidence_ratio",
            "dtw_line_coverage",
        ):
            if not isinstance(row.get(num_key), (int, float)):
                raise ValueError(f"payload.rows[{idx}].{num_key} must be numeric")
        for list_key in ("actions", "mismatch_examples", "suggested_targets"):
            val = row.get(list_key)
            if not isinstance(val, list) or not all(
                isinstance(item, str) for item in val
            ):
                raise ValueError(
                    f"payload.rows[{idx}].{list_key} must be an array of strings"
                )


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Human Guidance Task Recommendations", ""]
    lines.append(f"- Songs considered: `{payload.get('song_count_considered', 0)}`")
    lines.append(f"- Top limit: `{payload.get('top', 0)}`")
    lines.append(
        f"- Min priority: `{float(payload.get('min_priority', 0.0) or 0.0):.3f}`"
    )
    lines.append("")
    lines.append(
        "| Song | Priority | Agreement Cov | Agreement P95 (s) | Low Conf | DTW Line Cov |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in payload.get("rows", []) or []:
        lines.append(
            f"| {row.get('song', '')} | {float(row.get('priority_score', 0.0)):.3f} | "
            f"{float(row.get('agreement_coverage_ratio', 0.0)):.3f} | "
            f"{float(row.get('agreement_start_p95_abs_sec', 0.0)):.3f} | "
            f"{float(row.get('low_confidence_ratio', 0.0)):.3f} | "
            f"{float(row.get('dtw_line_coverage', 0.0)):.3f} |"
        )
        actions = row.get("actions", []) or []
        for action in actions:
            lines.append(f"- {row.get('song', '')}: {action}")
        mismatch_examples = row.get("mismatch_examples", []) or []
        for example in mismatch_examples:
            lines.append(f"- {row.get('song', '')}: example mismatch {example}")
        suggested_targets = row.get("suggested_targets", []) or []
        for target in suggested_targets:
            lines.append(f"- {row.get('song', '')}: suggested target {target}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ready_to_edit(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# song\tpriority\ttarget"]
    for row in payload.get("rows", []) or []:
        song = str(row.get("song") or "")
        priority = float(row.get("priority_score", 0.0) or 0.0)
        targets = row.get("suggested_targets", []) or []
        if not targets:
            continue
        for target in targets:
            lines.append(f"{song}\t{priority:.3f}\t{target}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, required=True, help="Run dir or report path"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="How many songs to include (0 means all)",
    )
    parser.add_argument(
        "--min-priority",
        type=float,
        default=0.0,
        help="Only include songs with priority score >= this value",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None, help="Output JSON path"
    )
    parser.add_argument(
        "--output-md", type=Path, default=None, help="Output markdown path"
    )
    parser.add_argument(
        "--output-ready-edit",
        type=Path,
        default=None,
        help="Optional compact TSV-like export for manual editing handoff",
    )
    parser.add_argument(
        "--validate-schema",
        action="store_true",
        help="Validate output JSON payload shape before writing artifacts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = args.report.expanduser().resolve()
    report = _load_report(report_path)
    payload = _recommend(
        report, top=int(args.top), min_priority=float(args.min_priority)
    )
    if bool(args.validate_schema):
        _validate_payload_schema(payload)

    out_root = report_path if report_path.is_dir() else report_path.parent
    out_json = args.output_json or (out_root / "human_guidance_tasks.json")
    out_md = args.output_md or (out_root / "human_guidance_tasks.md")
    out_ready = args.output_ready_edit
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown(out_md, payload)
    if out_ready is not None:
        _write_ready_to_edit(out_ready, payload)
    print("human_guidance_tasks: OK")
    print(f"  rows={len(payload.get('rows', []))}")
    print(f"  output_json={out_json}")
    print(f"  output_md={out_md}")
    if out_ready is not None:
        print(f"  output_ready_edit={out_ready}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
