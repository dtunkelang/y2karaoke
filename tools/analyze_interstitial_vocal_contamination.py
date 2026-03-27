"""Analyze inter-line windows for echo/backing-vocal contamination."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Sequence

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


@dataclass(frozen=True)
class GapWindow:
    index: int
    prev_text: str
    next_text: str
    start: float
    end: float


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _normalize_text(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def _token_overlap(a: str, b: str) -> float:
    a_tokens = set(_TOKEN_RE.findall(a.lower()))
    b_tokens = set(_TOKEN_RE.findall(b.lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _build_gap_windows(
    gold_lines: Sequence[dict[str, Any]],
    *,
    pad_before: float = 0.5,
    pad_after: float = 0.5,
    min_duration: float = 1.0,
) -> list[GapWindow]:
    windows: list[GapWindow] = []
    for idx in range(len(gold_lines) - 1):
        prev_line = gold_lines[idx]
        next_line = gold_lines[idx + 1]
        start = max(0.0, float(prev_line["end"]) - pad_before)
        end = max(start + min_duration, float(next_line["start"]) + pad_after)
        windows.append(
            GapWindow(
                index=idx + 1,
                prev_text=str(prev_line["text"]),
                next_text=str(next_line["text"]),
                start=round(start, 3),
                end=round(end, 3),
            )
        )
    return windows


def _trim_audio(source_audio: str, window: GapWindow, output_dir: Path) -> Path:
    import subprocess

    out = output_dir / f"gap_{window.index}_{window.index + 1}.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            source_audio,
            "-ss",
            f"{window.start:.3f}",
            "-t",
            f"{window.end - window.start:.3f}",
            str(out),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out


def _transcribe_variant(
    *,
    model: WhisperModel,
    audio_path: str,
    aggressive: bool,
) -> tuple[list[TranscriptionSegment], list[TranscriptionWord], str]:
    raw_segments, info = _run_whisper_transcription(
        model=model,
        vocals_path=audio_path,
        language=None,
        aggressive=aggressive,
        temperature=0.0,
    )
    segments, words = _convert_whisper_segments(raw_segments)
    return segments, words, str(info.language)


def _segment_text(segments: Sequence[TranscriptionSegment]) -> str:
    return " ".join(str(segment.text).strip() for segment in segments).strip()


def _classify_gap(
    *,
    prev_text: str,
    next_text: str,
    default_segments: Sequence[TranscriptionSegment],
    aggressive_segments: Sequence[TranscriptionSegment],
) -> str:
    aggressive_text = _segment_text(aggressive_segments)
    prev_overlap = _token_overlap(aggressive_text, prev_text)
    next_overlap = _token_overlap(aggressive_text, next_text)
    if default_segments:
        if max(prev_overlap, next_overlap) >= 0.5:
            return "supported_neighbor"
        return "unclear_supported"
    if not aggressive_segments:
        return "no_detectable_speech"
    if len(aggressive_segments) == 1:
        aggressive_duration = float(aggressive_segments[0].end) - float(
            aggressive_segments[0].start
        )
        if aggressive_duration >= 1.5 and max(prev_overlap, next_overlap) < 0.34:
            return "hallucinated_interstitial"
    if max(prev_overlap, next_overlap) >= 0.34:
        return "echo_fragment"
    return "unclear_interstitial"


def _analyze_gap(
    *,
    window: GapWindow,
    default_segments: Sequence[TranscriptionSegment],
    aggressive_segments: Sequence[TranscriptionSegment],
    default_language: str,
    aggressive_language: str,
) -> dict[str, Any]:
    default_text = _segment_text(default_segments)
    aggressive_text = _segment_text(aggressive_segments)
    return {
        "gap_index": window.index,
        "prev_text": window.prev_text,
        "next_text": window.next_text,
        "start": window.start,
        "end": window.end,
        "default_language": default_language,
        "aggressive_language": aggressive_language,
        "default_segment_count": len(default_segments),
        "aggressive_segment_count": len(aggressive_segments),
        "default_text": default_text,
        "aggressive_text": aggressive_text,
        "prev_overlap": round(_token_overlap(aggressive_text, window.prev_text), 3),
        "next_overlap": round(_token_overlap(aggressive_text, window.next_text), 3),
        "classification": _classify_gap(
            prev_text=window.prev_text,
            next_text=window.next_text,
            default_segments=default_segments,
            aggressive_segments=aggressive_segments,
        ),
    }


def analyze_gold_json(
    gold_json: Path,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    gold = _load_json(gold_json)
    audio_path = str(Path(gold["audio_path"]).expanduser().resolve())
    gold_lines = gold.get("lines", [])
    windows = _build_gap_windows(gold_lines)
    output_dir = output_dir or Path("/tmp/interstitial_windows")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")

    results = []
    for window in windows:
        clip_path = _trim_audio(audio_path, window, output_dir)
        default_segments, _default_words, default_language = _transcribe_variant(
            model=model,
            audio_path=str(clip_path),
            aggressive=False,
        )
        aggressive_segments, _aggressive_words, aggressive_language = (
            _transcribe_variant(
                model=model,
                audio_path=str(clip_path),
                aggressive=True,
            )
        )
        results.append(
            _analyze_gap(
                window=window,
                default_segments=default_segments,
                aggressive_segments=aggressive_segments,
                default_language=default_language,
                aggressive_language=aggressive_language,
            )
        )
    return {
        "gold_json": str(gold_json.resolve()),
        "audio_path": audio_path,
        "gaps": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gold_json", help="Gold timing JSON path")
    parser.add_argument(
        "--output-dir",
        default="/tmp/interstitial_windows",
        help="Directory for temporary trimmed gap clips",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    # Keep CLI-compatible output while exposing reusable analysis for other tools.
    payload = analyze_gold_json(
        Path(args.gold_json),
        output_dir=Path(args.output_dir),
    )
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"# {payload['gold_json']}")
    for gap in payload["gaps"]:
        print(
            f"## gap {gap['gap_index']}->{gap['gap_index'] + 1} "
            f"{gap['classification']}"
        )
        print(f"- prev: {gap['prev_text']}")
        print(f"- next: {gap['next_text']}")
        print(f"- aggressive: {gap['aggressive_text']}")
        print(
            f"- overlap prev/next: {gap['prev_overlap']:.3f}/{gap['next_overlap']:.3f}"
        )
        print(
            f"- default/aggressive segments: "
            f"{gap['default_segment_count']}/{gap['aggressive_segment_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
