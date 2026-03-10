#!/usr/bin/env python3
"""Trace the live pre-Whisper lyrics path used by get_lyrics_with_quality."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from y2karaoke.core.components.lyrics.helpers import (  # noqa: E402
    _apply_timing_to_lines,
)  # noqa: E402
from y2karaoke.core.components.lyrics.lyrics_whisper import (  # noqa: E402
    _detect_offset_with_issues,
    _refine_timing_with_quality,
)
from y2karaoke.core.components.lyrics.lyrics_whisper_quality import (  # noqa: E402
    _resolve_lrc_inputs,
)
from y2karaoke.core.components.lyrics.lrc import create_lines_from_lrc  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--title", required=True)
    p.add_argument("--artist", required=True)
    p.add_argument("--vocals-path", required=True)
    p.add_argument("--target-duration", type=int, default=None)
    p.add_argument("--line-from", type=int, default=22)
    p.add_argument("--line-to", type=int, default=27)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--json-out", type=Path, required=True)
    p.add_argument("--md-out", type=Path, required=True)
    return p.parse_args()


def _snapshot(
    stage: str, lines: list[Any], start_index: int, end_index: int, note: str = ""
) -> dict[str, Any]:
    rows = []
    for idx, line in enumerate(lines, start=1):
        if idx < start_index or idx > end_index or not line.words:
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
    return {"stage": stage, "count": len(lines), "note": note, "lines": rows}


def _to_md(data: dict[str, Any]) -> str:
    out = [f"# Live Pre-Whisper Trace: {data['artist']} - {data['title']}", ""]
    out.append(f"- Source: `{data['source']}`")
    out.append(f"- Offset applied: `{data['offset_applied_sec']:.2f}s`")
    out.append(f"- Issues: `{'; '.join(data['issues']) if data['issues'] else 'none'}`")
    out.append("")
    for stage in data["stages"]:
        out.append(f"## {stage['stage']}")
        if stage.get("note"):
            out.append(f"- Note: {stage['note']}")
        out.append(f"- Line count: `{stage['count']}`")
        out.append("")
        out.append("| # | Start | End | Dur | Text |")
        out.append("|---|---:|---:|---:|---|")
        for line in stage["lines"]:
            out.append(
                f"| {line['line_index']} | {line['start']:.3f} | {line['end']:.3f} | {line['duration']:.3f} | {line['text']} |"
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

    lrc_text, line_timings, source, file_lines = _resolve_lrc_inputs(
        title=args.title,
        artist=args.artist,
        lyrics_file=None,
        filter_promos=True,
        target_duration=args.target_duration,
        vocals_path=args.vocals_path,
        evaluate_sources=False,
        quality_report=quality_report,
        issues_list=issues,
        drop_lrc_line_timings=False,
        use_whisper=True,
        whisper_map_lrc=False,
        offline=args.offline,
    )
    if not (lrc_text and line_timings):
        raise SystemExit("No LRC timings available for trace")

    base_lines = create_lines_from_lrc(
        lrc_text,
        romanize=False,
        title=args.title,
        artist=args.artist,
        filter_promos=True,
    )
    _apply_timing_to_lines(base_lines, line_timings)

    adjusted_timings, offset = _detect_offset_with_issues(
        args.vocals_path,
        line_timings,
        None,
        issues,
    )
    offset_lines = create_lines_from_lrc(
        lrc_text,
        romanize=False,
        title=args.title,
        artist=args.artist,
        filter_promos=True,
    )
    _apply_timing_to_lines(offset_lines, adjusted_timings)

    refined_lines, method = _refine_timing_with_quality(
        offset_lines,
        args.vocals_path,
        adjusted_timings,
        lrc_text,
        args.target_duration,
        issues,
    )

    line_from = args.line_from
    line_to = args.line_to
    data = {
        "title": args.title,
        "artist": args.artist,
        "source": source,
        "offset_applied_sec": float(offset),
        "issues": issues,
        "alignment_method": method,
        "stages": [
            _snapshot("after_apply_timing", base_lines, line_from, line_to),
            _snapshot("after_offset", offset_lines, line_from, line_to),
            _snapshot(
                "after_pre_whisper_refine",
                refined_lines,
                line_from,
                line_to,
                note=method,
            ),
        ],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    args.md_out.write_text(_to_md(data), encoding="utf-8")
    print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
