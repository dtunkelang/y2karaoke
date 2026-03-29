"""Helpers for evaluating advisory support from alternate transcription variants."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Sequence

from ..alignment.timing_models import TranscriptionSegment, TranscriptionWord

_TOKEN_RE = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class AdvisoryLineSupport:
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


def normalize_support_text(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def token_overlap_score(a: str, b: str) -> float:
    a_tokens = set(_TOKEN_RE.findall(a.lower()))
    b_tokens = set(_TOKEN_RE.findall(b.lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def window_words(
    words: Iterable[TranscriptionWord],
    *,
    start: float,
    end: float,
) -> list[TranscriptionWord]:
    return [word for word in words if start <= float(word.start) < end]


def best_segment_text_match(
    segments: Sequence[TranscriptionSegment],
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
        score = token_overlap_score(line_text, segment.text)
        if score > best_score:
            best_score = score
            best_text = segment.text
    return best_text, round(best_score, 3)


def summarize_line_support(
    *,
    report_lines: Sequence[dict[str, Any]],
    default_segments: Sequence[TranscriptionSegment],
    default_words: Sequence[TranscriptionWord],
    aggressive_segments: Sequence[TranscriptionSegment],
    aggressive_words: Sequence[TranscriptionWord],
) -> list[AdvisoryLineSupport]:
    summaries: list[AdvisoryLineSupport] = []
    for idx, line in enumerate(report_lines):
        start = float(line["start"]) - 1.0
        next_start = (
            float(report_lines[idx + 1]["start"])
            if idx + 1 < len(report_lines)
            else float(line["end"]) + 2.0
        )
        default_window = window_words(default_words, start=start, end=next_start)
        aggressive_window = window_words(aggressive_words, start=start, end=next_start)
        default_seg_text, default_seg_score = best_segment_text_match(
            default_segments,
            line["text"],
            start=start,
            end=next_start,
        )
        aggressive_seg_text, aggressive_seg_score = best_segment_text_match(
            aggressive_segments,
            line["text"],
            start=start,
            end=next_start,
        )
        summaries.append(
            AdvisoryLineSupport(
                index=int(line["index"]),
                text=str(line["text"]),
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


def advisory_candidate_bucket(summary: AdvisoryLineSupport) -> str | None:
    if (
        summary.aggressive_best_overlap >= 0.95
        and summary.aggressive_window_word_count >= 3
        and summary.default_window_word_count == 0
    ):
        return "high_confidence"
    if (
        summary.aggressive_best_overlap >= 0.75
        and summary.aggressive_window_word_count >= 3
        and summary.current_window_word_count <= 3
    ):
        return "medium_confidence"
    if (
        summary.aggressive_best_overlap >= 0.99
        and summary.aggressive_window_word_count >= 3
        and summary.current_window_word_count <= 4
        and summary.default_best_overlap <= 0.2
    ):
        return "medium_confidence"
    return None


def advisory_candidate_score(summary: AdvisoryLineSupport) -> float:
    return (
        float(summary.aggressive_best_overlap) * 10.0
        + min(int(summary.aggressive_window_word_count), 8) * 0.2
        - min(int(summary.current_window_word_count), 8) * 0.15
    )
