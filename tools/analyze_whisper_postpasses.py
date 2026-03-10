#!/usr/bin/env python3
"""Trace key Whisper DTW postpasses for a song and capture line movement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from y2karaoke.core.components.lyrics.helpers import (  # noqa: E402
    _apply_timing_to_lines,
)  # noqa: E402
from y2karaoke.core.components.lyrics.lrc import create_lines_from_lrc  # noqa: E402
from y2karaoke.core.components.lyrics.lyrics_whisper import (  # noqa: E402
    _detect_offset_with_issues,
    _refine_timing_with_quality,
)
from y2karaoke.core.components.lyrics.lyrics_whisper_quality import (  # noqa: E402
    _resolve_lrc_inputs,
)
from y2karaoke.core.components.whisper import whisper_integration  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--title", required=True)
    p.add_argument("--artist", required=True)
    p.add_argument("--vocals-path", required=True)
    p.add_argument("--target-duration", type=int, default=None)
    p.add_argument("--line-from", type=int, default=23)
    p.add_argument("--line-to", type=int, default=26)
    p.add_argument("--json-out", type=Path, required=True)
    p.add_argument("--md-out", type=Path, required=True)
    return p.parse_args()


def _line_rows(lines: list[Any], line_from: int, line_to: int) -> list[dict[str, Any]]:
    rows = []
    for idx, line in enumerate(lines, start=1):
        if idx < line_from or idx > line_to or not line.words:
            continue
        rows.append(
            {
                "line_index": idx,
                "text": line.text,
                "start": round(line.start_time, 3),
                "end": round(line.end_time, 3),
                "duration": round(line.end_time - line.start_time, 3),
            }
        )
    return rows


def _to_md(payload: dict[str, Any]) -> str:
    out = [f"# Whisper Postpass Trace: {payload['artist']} - {payload['title']}", ""]
    out.append(f"- Alignment corrections: `{payload['alignment_count']}`")
    out.append("")
    for snap in payload["snapshots"]:
        out.append(f"## {snap['stage']}")
        out.append(f"- Line count: `{snap['count']}`")
        if snap.get("note"):
            out.append(f"- Note: {snap['note']}")
        out.append("")
        out.append("| # | Start | End | Dur | Text |")
        out.append("|---|---:|---:|---:|---|")
        for row in snap["lines"]:
            out.append(
                f"| {row['line_index']} | {row['start']:.3f} | {row['end']:.3f} | {row['duration']:.3f} | {row['text']} |"
            )
        out.append("")
    return "\n".join(out)


def main() -> int:
    args = _parse_args()
    issues: list[str] = []
    quality_report = {
        "lyrics_source_audio_scoring_used": False,
        "lyrics_source_disagreement_flagged": False,
        "lyrics_source_disagreement_reasons": [],
        "lyrics_source_candidate_count": 0,
        "lyrics_source_comparable_candidate_count": 0,
        "lyrics_source_selection_mode": "default",
        "lyrics_source_routing_skip_reason": "none",
        "source": "",
    }
    lrc_text, line_timings, _source, _file_lines = _resolve_lrc_inputs(
        title=args.title,
        artist=args.artist,
        lyrics_file=None,
        filter_promos=True,
        target_duration=args.target_duration,
        vocals_path=args.vocals_path,
        evaluate_sources=False,
        offline=False,
        quality_report=quality_report,
        issues_list=issues,
        drop_lrc_line_timings=False,
        use_whisper=True,
        whisper_map_lrc=False,
    )
    if not (lrc_text and line_timings):
        raise SystemExit("No LRC timings available")

    lines = create_lines_from_lrc(
        lrc_text,
        romanize=False,
        title=args.title,
        artist=args.artist,
        filter_promos=True,
    )
    adjusted_timings, _ = _detect_offset_with_issues(
        args.vocals_path,
        line_timings,
        None,
        issues,
    )
    _apply_timing_to_lines(lines, adjusted_timings)
    lines, _method = _refine_timing_with_quality(
        lines,
        args.vocals_path,
        adjusted_timings,
        lrc_text,
        args.target_duration,
        issues,
    )

    snapshots: list[dict[str, Any]] = [
        {
            "stage": "pre_whisper_input",
            "count": len(lines),
            "lines": _line_rows(lines, args.line_from, args.line_to),
        }
    ]

    names = [
        "_retime_adjacent_lines_to_whisper_window",
        "_retime_adjacent_lines_to_segment_window",
        "_pull_next_line_into_segment_window",
        "_pull_lines_near_segment_end",
        "_pull_next_line_into_same_segment",
        "_merge_lines_to_whisper_segments",
        "_tighten_lines_to_whisper_segments",
        "_pull_lines_to_best_segments",
    ]
    previous: dict[str, Any] = {}

    def wrap(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def _wrapped(*a: Any, **kw: Any) -> Any:
            result = fn(*a, **kw)
            lines_out = result[0] if isinstance(result, tuple) else result
            note = None
            if (
                isinstance(result, tuple)
                and len(result) > 1
                and isinstance(result[1], int)
            ):
                note = f"count={result[1]}"
            snapshots.append(
                {
                    "stage": name,
                    "count": len(lines_out),
                    "note": note,
                    "lines": _line_rows(lines_out, args.line_from, args.line_to),
                }
            )
            return result

        return _wrapped

    try:
        for name in names:
            previous[name] = getattr(whisper_integration, name)
            setattr(whisper_integration, name, wrap(name, previous[name]))

        aligned_lines, alignments, metrics = (
            whisper_integration.align_lrc_text_to_whisper_timings(
                lines,
                args.vocals_path,
                model_size="large",
                aggressive=False,
                temperature=0.0,
            )
        )
    finally:
        for name, fn in previous.items():
            setattr(whisper_integration, name, fn)

    snapshots.append(
        {
            "stage": "final_output",
            "count": len(aligned_lines),
            "note": f"metrics_keys={len(metrics or {})}",
            "lines": _line_rows(aligned_lines, args.line_from, args.line_to),
        }
    )

    payload = {
        "title": args.title,
        "artist": args.artist,
        "issues": issues,
        "alignment_count": len(alignments),
        "snapshots": snapshots,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.md_out.write_text(_to_md(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
