"""Compare transcription variants for a vocals clip."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from y2karaoke.core.components.whisper.whisper_integration_transcribe import (
    _convert_whisper_segments,
    _run_whisper_transcription,
    _transcribe_with_whisperx,
)

logger = logging.getLogger("analyze_transcription_variants")


@dataclass(frozen=True)
class VariantSummary:
    name: str
    language: str
    segment_count: int
    word_count: int
    transcript_end: float
    max_segment_duration: float
    max_segment_ratio: float
    dominant_single_segment: bool
    segments: list[dict[str, Any]]


def _summarize_variant(
    *,
    name: str,
    language: str,
    segments: list[Any],
    word_count: int,
    text_attr: str,
    limit: int = 6,
) -> VariantSummary:
    transcript_end = max((float(seg.end) for seg in segments), default=0.0)
    max_segment_duration = max(
        (max(0.0, float(seg.end) - float(seg.start)) for seg in segments),
        default=0.0,
    )
    max_segment_ratio = (
        max_segment_duration / transcript_end if transcript_end > 0.0 else 0.0
    )
    return VariantSummary(
        name=name,
        language=language,
        segment_count=len(segments),
        word_count=word_count,
        transcript_end=round(transcript_end, 3),
        max_segment_duration=round(max_segment_duration, 3),
        max_segment_ratio=round(max_segment_ratio, 3),
        dominant_single_segment=(
            len(segments) == 1 and transcript_end > 0.0 and max_segment_ratio > 0.75
        ),
        segments=[
            {
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "text": str(getattr(seg, text_attr)).strip(),
            }
            for seg in segments[:limit]
        ],
    )


def _run_faster_whisper_variant(
    *, model: WhisperModel, audio_path: str, aggressive: bool
) -> VariantSummary:
    raw_segments, info = _run_whisper_transcription(
        model=model,
        vocals_path=audio_path,
        language=None,
        aggressive=aggressive,
        temperature=0.0,
    )
    segments, words = _convert_whisper_segments(raw_segments)
    return _summarize_variant(
        name=f"faster_whisper:{'aggressive' if aggressive else 'default'}",
        language=str(info.language),
        segments=segments,
        word_count=len(words),
        text_attr="text",
    )


def _run_whisperx_variant(*, audio_path: str) -> VariantSummary | None:
    result = _transcribe_with_whisperx(
        vocals_path=audio_path,
        language=None,
        model_size="large",
        temperature=0.0,
        logger=logger,
    )
    if result is None:
        return None
    segments, words, language = result
    return _summarize_variant(
        name="whisperx_fallback",
        language=language,
        segments=segments,
        word_count=len(words),
        text_attr="text",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="Path to vocals audio clip")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of Markdown-ish text",
    )
    args = parser.parse_args()

    audio_path = str(Path(args.audio).expanduser().resolve())
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")

    variants = [
        _run_faster_whisper_variant(
            model=model,
            audio_path=audio_path,
            aggressive=False,
        ),
        _run_faster_whisper_variant(
            model=model,
            audio_path=audio_path,
            aggressive=True,
        ),
    ]
    whisperx_summary = _run_whisperx_variant(audio_path=audio_path)
    if whisperx_summary is not None:
        variants.append(whisperx_summary)

    payload = {
        "audio": audio_path,
        "variants": [asdict(variant) for variant in variants],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"# {audio_path}")
    for variant in variants:
        print(f"## {variant.name}")
        print(
            f"- language={variant.language} segments={variant.segment_count} "
            f"words={variant.word_count} end={variant.transcript_end:.3f}"
        )
        print(
            f"- max_segment_duration={variant.max_segment_duration:.3f} "
            f"max_segment_ratio={variant.max_segment_ratio:.3f} "
            f"dominant_single_segment={variant.dominant_single_segment}"
        )
        for segment in variant.segments:
            print(f"- {segment['start']:.3f}->{segment['end']:.3f} | {segment['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
