"""Compare line-level support under default and aggressive transcription windows."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from y2karaoke.core.components.alignment.timing_models import (
    TranscriptionSegment,
    TranscriptionWord,
)
from y2karaoke.core.components.whisper.whisper_integration_transcribe import (
    _convert_whisper_segments,
    _run_whisper_transcription,
)

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _normalize_text(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def _token_set(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _token_overlap_score(a: str, b: str) -> float:
    a_tokens = _token_set(a)
    b_tokens = _token_set(b)
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


@dataclass(frozen=True)
class LineSupportSummary:
    index: int
    text: str
    current_window_word_count: int
    default_window_word_count: int
    aggressive_window_word_count: int
    default_best_segment_text: str
    aggressive_best_segment_text: str
    default_best_overlap: float
    aggressive_best_overlap: float
    aggressive_gain: bool


def _window_words(
    words: list[TranscriptionWord],
    *,
    start: float,
    end: float,
) -> list[TranscriptionWord]:
    return [word for word in words if start <= float(word.start) < end]


def _best_segment(
    segments: list[TranscriptionSegment],
    line_text: str,
    *,
    start: float,
    end: float,
) -> tuple[str, float]:
    best_text = ""
    best_score = 0.0
    for segment in segments:
        seg_start = float(segment.start)
        seg_end = float(segment.end)
        if seg_end < start - 1.0 or seg_start > end + 1.0:
            continue
        score = _token_overlap_score(line_text, segment.text)
        if score > best_score:
            best_score = score
            best_text = segment.text
    return best_text, round(best_score, 3)


def _load_report(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _transcribe_variant(
    *,
    model: WhisperModel,
    audio_path: str,
    aggressive: bool,
) -> tuple[list[TranscriptionSegment], list[TranscriptionWord]]:
    raw_segments, _info = _run_whisper_transcription(
        model=model,
        vocals_path=audio_path,
        language=None,
        aggressive=aggressive,
        temperature=0.0,
    )
    return _convert_whisper_segments(raw_segments)


def _summarize_lines(
    *,
    report: dict[str, Any],
    default_segments: list[TranscriptionSegment],
    default_words: list[TranscriptionWord],
    aggressive_segments: list[TranscriptionSegment],
    aggressive_words: list[TranscriptionWord],
) -> list[LineSupportSummary]:
    lines = report.get("lines", [])
    summaries: list[LineSupportSummary] = []
    for idx, line in enumerate(lines):
        start = float(line["start"]) - 1.0
        next_start = (
            float(lines[idx + 1]["start"])
            if idx + 1 < len(lines)
            else float(line["end"]) + 2.0
        )
        default_window = _window_words(default_words, start=start, end=next_start)
        aggressive_window = _window_words(aggressive_words, start=start, end=next_start)
        default_seg_text, default_seg_score = _best_segment(
            default_segments,
            line["text"],
            start=start,
            end=next_start,
        )
        aggressive_seg_text, aggressive_seg_score = _best_segment(
            aggressive_segments,
            line["text"],
            start=start,
            end=next_start,
        )
        summaries.append(
            LineSupportSummary(
                index=int(line["index"]),
                text=line["text"],
                current_window_word_count=int(line.get("whisper_window_word_count", 0)),
                default_window_word_count=len(default_window),
                aggressive_window_word_count=len(aggressive_window),
                default_best_segment_text=default_seg_text,
                aggressive_best_segment_text=aggressive_seg_text,
                default_best_overlap=default_seg_score,
                aggressive_best_overlap=aggressive_seg_score,
                aggressive_gain=(
                    len(aggressive_window) > len(default_window)
                    and aggressive_seg_score >= default_seg_score + 0.2
                ),
            )
        )
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timing_report", help="Timing report JSON path")
    parser.add_argument("audio", help="Clip vocals/audio path")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    report = _load_report(Path(args.timing_report))
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    default_segments, default_words = _transcribe_variant(
        model=model,
        audio_path=args.audio,
        aggressive=False,
    )
    aggressive_segments, aggressive_words = _transcribe_variant(
        model=model,
        audio_path=args.audio,
        aggressive=True,
    )
    summaries = _summarize_lines(
        report=report,
        default_segments=default_segments,
        default_words=default_words,
        aggressive_segments=aggressive_segments,
        aggressive_words=aggressive_words,
    )
    payload = {
        "timing_report": str(Path(args.timing_report).resolve()),
        "audio": str(Path(args.audio).resolve()),
        "lines": [asdict(summary) for summary in summaries],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"# {payload['timing_report']}")
    for line in summaries:
        print(f"## line {line.index} {line.text}")
        print(
            f"- current/default/aggressive window words: "
            f"{line.current_window_word_count}/{line.default_window_word_count}/{line.aggressive_window_word_count}"
        )
        print(
            f"- default overlap={line.default_best_overlap:.3f} "
            f"aggressive overlap={line.aggressive_best_overlap:.3f} "
            f"aggressive_gain={line.aggressive_gain}"
        )
        if line.default_best_segment_text:
            print(f"- default segment: {line.default_best_segment_text}")
        if line.aggressive_best_segment_text:
            print(f"- aggressive segment: {line.aggressive_best_segment_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
