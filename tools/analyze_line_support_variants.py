"""Compare line-level support under default and aggressive transcription windows."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from y2karaoke.core.components.alignment.timing_models import (
    TranscriptionSegment,
    TranscriptionWord,
)
from y2karaoke.core.components.whisper.whisper_advisory_support import (
    AdvisoryLineSupport,
    summarize_line_support,
)
from y2karaoke.core.components.whisper.whisper_integration_transcribe import (
    _convert_whisper_segments,
    _run_whisper_transcription,
)


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
) -> list[AdvisoryLineSupport]:
    return summarize_line_support(
        report_lines=report.get("lines", []),
        default_segments=default_segments,
        default_words=default_words,
        aggressive_segments=aggressive_segments,
        aggressive_words=aggressive_words,
    )


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
