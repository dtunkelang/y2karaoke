#!/usr/bin/env python3
"""Trace the audio-only LRC refinement stages for a song."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from y2karaoke.core.audio_analysis import (  # noqa: E402
    _check_for_silence_in_range,
    _check_vocal_activity_in_range,
    extract_audio_features,
)
from y2karaoke.core.components.alignment.alignment import (  # noqa: E402
    _apply_adjustments_to_lines,
    adjust_timing_for_duration_mismatch,
)
from y2karaoke.core.components.alignment.timing_evaluator_correction import (  # noqa: E402
    correct_line_timestamps,
    fix_spurious_gaps,
)
from y2karaoke.core.components.lyrics.helpers import (  # noqa: E402
    _apply_duration_mismatch_adjustment,
    _apply_timing_to_lines,
    _compress_spurious_lrc_gaps,
    _detect_and_apply_offset,
)
from y2karaoke.core.components.lyrics.lrc import (  # noqa: E402
    create_lines_from_lrc,
    parse_lrc_with_timing,
)
from y2karaoke.core.components.lyrics.sync import get_lrc_duration  # noqa: E402
from y2karaoke.core.components.whisper.whisper_alignment_refinement import (  # noqa: E402
    _pull_lines_forward_for_continuous_vocals,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace the raw-LRC audio refinement stages and capture line geometry "
            "after each step."
        )
    )
    parser.add_argument("--title", required=True, help="Song title")
    parser.add_argument("--artist", required=True, help="Artist name")
    parser.add_argument(
        "--vocals-path",
        required=True,
        help="Path to vocals or mixed audio used for refinement",
    )
    parser.add_argument(
        "--lyrics-cache",
        type=Path,
        default=Path.home() / ".cache" / "karaoke" / "lyrics_cache.json",
        help="Path to karaoke lyrics cache JSON",
    )
    parser.add_argument(
        "--target-duration",
        type=int,
        default=203,
        help="Expected target duration in seconds",
    )
    parser.add_argument(
        "--line-from",
        type=int,
        default=1,
        help="1-based first line to include in stage snapshots",
    )
    parser.add_argument(
        "--line-to",
        type=int,
        default=9999,
        help="1-based last line to include in stage snapshots",
    )
    parser.add_argument("--json-out", type=Path, help="Optional JSON output path")
    parser.add_argument("--md-out", type=Path, help="Optional Markdown output path")
    return parser.parse_args()


def _cache_key(artist: str, title: str) -> str:
    return f"{artist.strip().lower()}|{title.strip().lower()}"


def _load_lrc_text(cache_path: Path, artist: str, title: str) -> str:
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    entry = raw["lrc_cache"][_cache_key(artist, title)]
    if not isinstance(entry, list) or not entry:
        raise KeyError(f"No LRC cache entry for {artist} - {title}")
    lrc_text = entry[0]
    if not isinstance(lrc_text, str):
        raise ValueError(f"Unexpected LRC cache payload for {artist} - {title}")
    return lrc_text


def _serialize_lines(
    lines: list[Any], line_from: int, line_to: int
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, line in enumerate(lines, start=1):
        if idx < line_from or idx > line_to:
            continue
        out.append(
            {
                "line_index": idx,
                "text": line.text,
                "start": round(float(line.start_time), 3),
                "end": round(float(line.end_time), 3),
                "duration": round(float(line.end_time - line.start_time), 3),
            }
        )
    return out


def _to_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# LRC Refinement Trace: {payload['artist']} - {payload['title']}")
    lines.append("")
    lines.append(f"- Lyrics cache: `{payload['lyrics_cache']}`")
    lines.append(f"- Vocals path: `{payload['vocals_path']}`")
    lines.append(f"- Offset applied: `{payload['offset_applied_sec']:+.2f}s`")
    lines.append(f"- Parsed timed lines: `{payload['parsed_timed_line_count']}`")
    lines.append(f"- Initial lyric lines: `{payload['initial_line_count']}`")
    lines.append(f"- Final traced line count: `{payload['final_line_count']}`")
    lines.append("")
    for stage in payload["stages"]:
        lines.append(f"## {stage['stage']}")
        if stage.get("note"):
            lines.append(f"- Note: {stage['note']}")
        if "count" in stage:
            lines.append(f"- Line count: `{stage['count']}`")
        if "corrections" in stage:
            lines.append(f"- Corrections: `{stage['corrections']}`")
        if "fixes" in stage:
            lines.append(f"- Fixes: `{stage['fixes']}`")
        lines.append("")
        lines.append("| # | Start | End | Dur | Text |")
        lines.append("|---|---:|---:|---:|---|")
        for row in stage["lines"]:
            lines.append(
                f"| {row['line_index']} | {row['start']:.3f} | {row['end']:.3f} | {row['duration']:.3f} | {row['text']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    lrc_text = _load_lrc_text(args.lyrics_cache, args.artist, args.title)
    line_timings = parse_lrc_with_timing(lrc_text, args.title, args.artist, True)
    lines = create_lines_from_lrc(
        lrc_text,
        romanize=False,
        title=args.title,
        artist=args.artist,
        filter_promos=True,
    )
    _apply_timing_to_lines(lines, line_timings)

    line_timings, offset = _detect_and_apply_offset(
        args.vocals_path,
        line_timings,
        None,
    )
    lines = create_lines_from_lrc(
        lrc_text,
        romanize=False,
        title=args.title,
        artist=args.artist,
        filter_promos=True,
    )
    _apply_timing_to_lines(lines, line_timings)
    audio_features = extract_audio_features(args.vocals_path)
    if audio_features is None:
        raise RuntimeError("Could not extract audio features")

    stages: list[dict[str, Any]] = [
        {
            "stage": "after_apply_timing",
            "count": len(lines),
            "lines": _serialize_lines(lines, args.line_from, args.line_to),
        }
    ]

    duration_lines = _apply_duration_mismatch_adjustment(
        deepcopy(lines),
        line_timings,
        args.vocals_path,
        lrc_text=lrc_text,
        target_duration=args.target_duration,
        get_lrc_duration=get_lrc_duration,
        adjust_timing_for_duration_mismatch=adjust_timing_for_duration_mismatch,
    )
    stages.append(
        {
            "stage": "after_duration_mismatch_adjustment",
            "count": len(duration_lines),
            "lines": _serialize_lines(duration_lines, args.line_from, args.line_to),
        }
    )

    compressed_lines, compressed_count = _compress_spurious_lrc_gaps(
        deepcopy(duration_lines),
        line_timings,
        audio_features,
        _apply_adjustments_to_lines,
        _check_vocal_activity_in_range,
        _check_for_silence_in_range,
    )
    stages.append(
        {
            "stage": "after_spurious_gap_compress",
            "count": len(compressed_lines),
            "fixes": compressed_count,
            "lines": _serialize_lines(compressed_lines, args.line_from, args.line_to),
        }
    )

    corrected_lines, corrections = correct_line_timestamps(
        deepcopy(compressed_lines),
        audio_features,
        max_correction=3.0,
    )
    stages.append(
        {
            "stage": "after_correct_line_timestamps",
            "count": len(corrected_lines),
            "corrections": len(corrections),
            "note": "; ".join(corrections[:5]) if corrections else "",
            "lines": _serialize_lines(corrected_lines, args.line_from, args.line_to),
        }
    )

    pulled_lines, pull_count = _pull_lines_forward_for_continuous_vocals(
        deepcopy(corrected_lines),
        audio_features,
    )
    stages.append(
        {
            "stage": "after_pull_lines_forward_for_continuous_vocals",
            "count": len(pulled_lines),
            "fixes": pull_count,
            "lines": _serialize_lines(pulled_lines, args.line_from, args.line_to),
        }
    )

    merged_lines, gap_fixes = fix_spurious_gaps(deepcopy(pulled_lines), audio_features)
    stages.append(
        {
            "stage": "after_fix_spurious_gaps",
            "count": len(merged_lines),
            "fixes": len(gap_fixes),
            "note": "; ".join(gap_fixes[:5]) if gap_fixes else "",
            "lines": _serialize_lines(merged_lines, args.line_from, args.line_to),
        }
    )

    payload = {
        "title": args.title,
        "artist": args.artist,
        "lyrics_cache": str(args.lyrics_cache),
        "vocals_path": str(args.vocals_path),
        "offset_applied_sec": round(float(offset), 4),
        "parsed_timed_line_count": len(line_timings),
        "initial_line_count": len(lines),
        "final_line_count": len(merged_lines),
        "stages": stages,
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(_to_markdown(payload), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
