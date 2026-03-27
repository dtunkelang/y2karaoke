"""Compare fuzzy segment-level support across transcription variants."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

from faster_whisper import WhisperModel

from y2karaoke.core import phonetic_utils
from y2karaoke.core.components.alignment.timing_models import (
    TranscriptionSegment,
    TranscriptionWord,
)
from y2karaoke.core.components.whisper.whisper_advisory_support import (
    token_overlap_score,
)
from y2karaoke.core.components.whisper.whisper_integration_transcribe import (
    _convert_whisper_segments,
    _run_whisper_transcription,
)

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _normalize_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _window_token_sequences(
    tokens: list[str], *, target_len: int
) -> list[tuple[int, int, list[str]]]:
    if not tokens or target_len <= 0:
        return []
    windows: list[tuple[int, int, list[str]]] = []
    min_len = max(1, target_len - 1)
    max_len = min(len(tokens), target_len + 1)
    for window_len in range(min_len, max_len + 1):
        for start in range(0, len(tokens) - window_len + 1):
            end = start + window_len
            windows.append((start, end, tokens[start:end]))
    return windows


def _window_joint_score(
    line_tokens: list[str], window_tokens: list[str], *, language: str
) -> tuple[float, float, float]:
    line_text = " ".join(line_tokens)
    window_text = " ".join(window_tokens)
    text_score = token_overlap_score(line_text, window_text)
    phonetic_scores = []
    for line_token, window_token in zip(line_tokens, window_tokens):
        phonetic_scores.append(
            phonetic_utils._phonetic_similarity(line_token, window_token, language)
        )
    phonetic_mean = (
        sum(phonetic_scores) / max(1, len(phonetic_scores)) if phonetic_scores else 0.0
    )
    joint_score = max(text_score, phonetic_mean)
    return text_score, phonetic_mean, joint_score


def _best_segment_window(
    *,
    line_text: str,
    segment_text: str,
    language: str,
) -> dict[str, Any]:
    line_tokens = _normalize_tokens(line_text)
    segment_tokens = _normalize_tokens(segment_text)
    best: dict[str, Any] = {
        "span_text": "",
        "span_start": None,
        "span_end": None,
        "text_score": 0.0,
        "phonetic_mean": 0.0,
        "joint_score": 0.0,
    }
    best_key = (-1.0, -1.0, -1.0)
    for start, end, window_tokens in _window_token_sequences(
        segment_tokens, target_len=len(line_tokens)
    ):
        text_score, phonetic_mean, joint_score = _window_joint_score(
            line_tokens,
            window_tokens,
            language=language,
        )
        candidate_key = (
            joint_score,
            text_score,
            -abs(len(window_tokens) - len(line_tokens)),
        )
        if candidate_key <= best_key:
            continue
        best_key = candidate_key
        best = {
            "span_text": " ".join(window_tokens),
            "span_start": start,
            "span_end": end,
            "text_score": round(text_score, 3),
            "phonetic_mean": round(phonetic_mean, 3),
            "joint_score": round(joint_score, 3),
        }
    return best


def _estimate_span_times(
    *,
    segment_start: float,
    segment_end: float,
    segment_text: str,
    span_start: int | None,
    span_end: int | None,
) -> tuple[float | None, float | None]:
    segment_tokens = _normalize_tokens(segment_text)
    if (
        span_start is None
        or span_end is None
        or not segment_tokens
        or span_start < 0
        or span_end > len(segment_tokens)
        or span_start >= span_end
    ):
        return None, None
    duration = max(0.0, segment_end - segment_start)
    if duration <= 0.0:
        return None, None
    token_count = len(segment_tokens)
    est_start = segment_start + duration * (span_start / token_count)
    est_end = segment_start + duration * (span_end / token_count)
    return round(est_start, 3), round(est_end, 3)


def _best_fuzzy_segment_match(
    *,
    segments: Sequence[TranscriptionSegment],
    line_text: str,
    line_start: float,
    line_end: float,
    language: str,
) -> dict[str, Any]:
    best: dict[str, Any] = {
        "segment_text": "",
        "segment_start": None,
        "segment_end": None,
        "span_text": "",
        "span_start": None,
        "span_end": None,
        "text_score": 0.0,
        "phonetic_mean": 0.0,
        "joint_score": 0.0,
    }
    for segment in segments:
        seg_start = float(segment.start)
        seg_end = float(segment.end)
        if seg_end < line_start - 1.0 or seg_start > line_end + 1.0:
            continue
        candidate = _best_segment_window(
            line_text=line_text,
            segment_text=segment.text,
            language=language,
        )
        if candidate["joint_score"] <= float(best["joint_score"]):
            continue
        best = {
            "segment_text": segment.text,
            "segment_start": seg_start,
            "segment_end": seg_end,
            **candidate,
        }
    est_start, est_end = _estimate_span_times(
        segment_start=float(best["segment_start"] or 0.0),
        segment_end=float(best["segment_end"] or 0.0),
        segment_text=str(best["segment_text"] or ""),
        span_start=best["span_start"],
        span_end=best["span_end"],
    )
    best["estimated_span_start"] = est_start
    best["estimated_span_end"] = est_end
    return best


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


def _analyze_report(
    *,
    report: dict[str, Any],
    default_segments: Sequence[TranscriptionSegment],
    aggressive_segments: Sequence[TranscriptionSegment],
    language: str,
    line_indexes: set[int],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for line in report.get("lines", []):
        index = int(line["index"])
        if line_indexes and index not in line_indexes:
            continue
        payload.append(
            {
                "index": index,
                "text": str(line["text"]),
                "current_window_text": " ".join(
                    str(word.get("text") or "")
                    for word in line.get("whisper_window_words", [])
                ).strip(),
                "default_best": _best_fuzzy_segment_match(
                    segments=default_segments,
                    line_text=str(line["text"]),
                    line_start=float(line["start"]),
                    line_end=float(line["end"]),
                    language=language,
                ),
                "aggressive_best": _best_fuzzy_segment_match(
                    segments=aggressive_segments,
                    line_text=str(line["text"]),
                    line_start=float(line["start"]),
                    line_end=float(line["end"]),
                    language=language,
                ),
            }
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timing_report", help="Timing report JSON path")
    parser.add_argument("audio", help="Clip audio path")
    parser.add_argument("--language", default="en", help="Phonetic language code")
    parser.add_argument(
        "--line",
        type=int,
        action="append",
        dest="line_indexes",
        help="Specific 1-based line index to analyze; repeatable",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    report = _load_json(Path(args.timing_report))
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    default_segments, _default_words = _transcribe_variant(
        model=model,
        audio_path=str(Path(args.audio).expanduser().resolve()),
        aggressive=False,
    )
    aggressive_segments, _aggressive_words = _transcribe_variant(
        model=model,
        audio_path=str(Path(args.audio).expanduser().resolve()),
        aggressive=True,
    )
    lines = _analyze_report(
        report=report,
        default_segments=default_segments,
        aggressive_segments=aggressive_segments,
        language=args.language,
        line_indexes=set(args.line_indexes or []),
    )
    payload = {
        "timing_report": str(Path(args.timing_report).resolve()),
        "audio": str(Path(args.audio).expanduser().resolve()),
        "language": args.language,
        "lines": lines,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"# {payload['timing_report']}")
    for line in lines:
        print(f"## line {line['index']} {line['text']}")
        print(f"- current window: {line['current_window_text']}")
        for label in ("default_best", "aggressive_best"):
            best = line[label]
            print(
                f"- {label}: {best['segment_text']} | span={best['span_text']} "
                f"(text={best['text_score']:.3f}, phon={best['phonetic_mean']:.3f}, "
                f"joint={best['joint_score']:.3f})"
            )
            if best["estimated_span_start"] is not None:
                print(
                    f"  est span time: {best['estimated_span_start']:.3f}-"
                    f"{best['estimated_span_end']:.3f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
